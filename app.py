from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Body
from fastapi.responses import FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps
from datetime import datetime
import io, base64, os, time, re, json, sqlite3, hashlib, csv, threading
import urllib.request
from urllib.parse import urljoin
import cv2
import numpy as np

app = FastAPI(title="Mother's Hut Meal Verification System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best.pt")
YOLOE_PROMPT_PATH = os.path.join(BASE_DIR, "yoloe-11s-seg.pt")
DB_PATH = os.path.join(BASE_DIR, "dispatch.db")

model = None
yoloe_prompt_model = None
_cached_prompt_classes = None
_prompt_lock = threading.Lock()

if os.path.exists(MODEL_PATH):
    from ultralytics import YOLO
    model = YOLO(MODEL_PATH)
    print(f"✅ Loaded custom model '{MODEL_PATH}' ({len(model.names)} classes)")
else:
    print("⚠️ best.pt not found — running custom model in Demo Mode")

try:
    from ultralytics import YOLOE
    p_src = YOLOE_PROMPT_PATH if os.path.exists(YOLOE_PROMPT_PATH) else "yoloe-11s-seg.pt"
    yoloe_prompt_model = YOLOE(p_src)
    print("✅ Loaded Hybrid YOLOE Dish Model (Text-Prompt Food Vocabulary)")
except Exception as e:
    print(f"⚠️ YOLOE open-vocab model unavailable ({e}) — falling back to custom model only")

DEMO_CLASSES = [
    "Aloo Gobi", "Biryani", "Butter Chicken", "Chana Masala",
    "Chapati", "Dal Makhani", "Gulab Jamun", "Jalebi", "Naan", "Palak Paneer"
]

# Strict whole-dish food vocabulary (ingredient-only words removed to prevent splitting one dish into parts)
FOOD_DISH_VOCAB = [
    "aloo gobi", "aloo paratha", "appam", "baked rasgulla", "bhatura", "bhature",
    "biryani", "boiled egg", "bread", "burger", "butter chicken",
    "chana masala", "chapati", "cheese omelette", "chicken curry",
    "chicken lollipop", "chicken steak", "chicken tikka", "chicken tikka masala",
    "chilli chicken", "chilli paneer", "chole bhature", "chutney", "coffee",
    "cold coffee", "cookies", "corn flakes", "crab", "cutlet",
    "dal makhani", "dal tadka", "doi", "dosa", "egg omelette",
    "egg poach", "fish cutlet", "fish finger", "fish fry", "french fries",
    "fried rice", "gulab jamun", "hakka noodles", "idli", "ilish", "jalebi",
    "juice", "kadai paneer", "kala jamun", "kebab", "keema", "khichdi",
    "kochuri", "kulcha", "lassi", "luchi", "lyangcha", "malai curry",
    "masala dosa", "medu vada", "milk", "milkshake", "mixed veg", "momos",
    "mushroom", "mutton curry", "naan", "nachos", "noodles",
    "omelette", "palak paneer", "paneer butter masala", "paneer tikka",
    "papad", "paratha", "pasta", "penne pasta", "pizza", "plain dosa",
    "pomfret", "prawn", "pulao", "raita", "rasgulla", "rasmalai",
    "roti", "salad", "sambar vada", "samosa", "sandesh", "sandwich",
    "seekh kebab", "shorbot", "soup", "spring roll", "steamed rice", "tea",
    "tandoori chicken", "tandoori roti", "thai curry", "thatte idli", "vada",
    "veg biryani"
]

GENERIC_FOOD_WORDS = {
    "veg", "non", "pc", "pcs", "half", "full", "masala", "curry",
    "fry", "tikka", "rice", "egg", "paneer", "dal", "chicken",
    "mutton", "fish", "butter", "gravy", "sauce", "special"
}


def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS dispatch_orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                display_time TEXT NOT NULL,
                expected_json TEXT NOT NULL,
                detected_json TEXT NOT NULL,
                fill_pct REAL NOT NULL,
                min_fill REAL NOT NULL,
                missing_json TEXT NOT NULL,
                status TEXT NOT NULL,
                proof_hash TEXT NOT NULL,
                payload_json TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS csv_store (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                filename TEXT NOT NULL,
                items_json TEXT NOT NULL
            )
        """)
        conn.commit()


def compute_proof_hash(
    order_id: str,
    timestamp: str,
    fill_pct: float,
    detected: list,
    status: str,
    img_b64: str
) -> str:
    raw = (
        f"{order_id}|{timestamp}|{fill_pct}|"
        f"{json.dumps(detected, sort_keys=True)}|{status}|{img_b64[:256]}"
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


init_db()


def to_base64(pil_img: Image.Image) -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=92)
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"


def draw_top_left_label(pil_img: Image.Image, detections: list) -> str:
    """Draw clean top-left confidence text matching the 18-dish classification output (no pink masks)."""
    bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]

    if detections:
        lines = [
            f"{d['name']} {d['confidence'] / 100.0:.2f}"
            for d in detections[:3]
        ]
    else:
        lines = ["No dish detected"]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.55, min(1.05, w / 720.0))
    thickness = max(1, int(round(font_scale * 2)))
    pad_x = 10
    pad_y = 6
    y = 14

    for text in lines:
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        x1, y1 = 12, y
        x2, y2 = min(w - 6, x1 + tw + pad_x * 2), min(h - 6, y1 + th + pad_y * 2)
        cv2.rectangle(bgr, (x1, y1), (x2, y2), (255, 255, 255), -1)
        cv2.putText(
            bgr,
            text,
            (x1 + pad_x, y1 + pad_y + th - 1),
            font,
            font_scale,
            (24, 24, 27),
            thickness,
            cv2.LINE_AA
        )
        y = y2 + 6

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return to_base64(Image.fromarray(rgb))


def estimate_fill_percentage(pil_img: Image.Image) -> float:
    """Estimate bowl/plate food fill percentage (0-100%) using color and texture."""
    small = np.array(pil_img.resize((240, 240)))
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32)

    mean = cv2.blur(gray, (9, 9))
    mean_sq = cv2.blur(gray ** 2, (9, 9))
    local_std = np.sqrt(np.maximum(mean_sq - mean ** 2, 0))

    h, w = gray.shape
    Y, X = np.ogrid[:h, :w]

    roi_mask = (
        np.sqrt((X - w / 2) ** 2 + (Y - h / 2) ** 2)
        <= (min(h, w) * 0.36)
    )

    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    rich_color = (s >= 42) & (v >= 30) & (v <= 245)
    textured_food = (
        (local_std >= 16) & (s >= 18) & (v >= 35) & (v <= 240)
    )

    food_mask = (rich_color | textured_food) & roi_mask

    total_roi = int(np.count_nonzero(roi_mask))
    food_pixels = int(np.count_nonzero(food_mask))

    raw_pct = (
        (food_pixels / total_roi) * 100.0
        if total_roi > 0 else 0.0
    )

    return float(min(100.0, round(float(raw_pct) * 1.05, 1)))


def get_trained_class_names() -> list:
    if model is not None:
        return [str(n).title() for n in model.names.values()]
    return DEMO_CLASSES


def clean_item_phrase(raw_item: str) -> str:
    """Turn '2252 - Boiled Egg 2pcs (with chutney)' into 'boiled egg'."""
    s = re.sub(r"^\d+\s*-\s*", "", raw_item.strip())
    s = re.sub(r"\([^)]*\)", " ", s)
    s = re.sub(r"\b\d+\s*(pcs?|pieces?|gm|g|kg|ml|ltr|l)\b", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"[^a-zA-Z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def matches_trained_menu(expected_item: str) -> str | None:
    cleaned = clean_item_phrase(expected_item)
    trained = sorted(get_trained_class_names(), key=len, reverse=True)
    for dish in trained:
        d = dish.lower().strip()
        if re.search(rf"(^|\s){re.escape(d)}(\s|$)", cleaned):
            return dish
    return None


def is_expected_matched(expected_item: str, detected_names: set) -> bool:
    exp_low = expected_item.lower().strip()
    det_lows = {d.lower().strip() for d in detected_names}
    if exp_low in det_lows:
        return True

    exp_clean = clean_item_phrase(expected_item)
    if not exp_clean:
        return False

    for det in det_lows:
        det_clean = clean_item_phrase(det)
        if not det_clean:
            continue
        if exp_clean == det_clean or exp_clean in det_clean or det_clean in exp_clean:
            return True
    return False


def _box_iou(b1: tuple, b2: tuple) -> float:
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0.0:
        return 0.0
    a1 = max(1e-6, (b1[2] - b1[0]) * (b1[3] - b1[1]))
    a2 = max(1e-6, (b2[2] - b2[0]) * (b2[3] - b2[1]))
    return inter / (a1 + a2 - inter)


def _strip_meta(detections: list) -> list:
    return [
        {"name": d["name"], "confidence": d["confidence"], "source": d["source"]}
        for d in detections
    ]


def extract_box_detections(
    result,
    fallback_names,
    source_label: str,
    img_size: tuple | None = None,
    include_meta: bool = False
) -> list:
    detections = []
    names_map = getattr(result, "names", None) or fallback_names
    img_area = float(max(1, img_size[0] * img_size[1])) if img_size else None

    if result.boxes is not None:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            raw_name = names_map[cls_id] if isinstance(names_map, (dict, list)) else str(cls_id)
            name = str(raw_name).strip().title()
            score = float(round(float(box.conf[0]) * 100, 1))
            item = {"name": name, "confidence": score, "source": source_label}
            if include_meta:
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                item["_bbox"] = (x1, y1, x2, y2)
                if img_area:
                    item["_area"] = max(0.0, (x2 - x1) * (y2 - y1)) / img_area
            detections.append(item)
        detections.sort(key=lambda x: x["confidence"], reverse=True)
    return detections


def run_yolo_on_image(
    image: Image.Image,
    conf_val: float,
    with_plot: bool = True,
    imgsz: int = 640,
    include_meta: bool = False
):
    """Run custom 18-dish YOLO model."""
    if model is None:
        dets = [{"name": "Butter Chicken", "confidence": 91.7, "source": "18-Dish AI"}]
        ann = draw_top_left_label(image, dets) if with_plot else ""
        return dets, ann, False

    results = model.predict(image, conf=conf_val, imgsz=imgsz, verbose=False)
    is_box_model = results[0].boxes is not None
    detections = []

    if is_box_model:
        detections = extract_box_detections(
            results[0],
            model.names,
            "18-Dish AI",
            img_size=image.size,
            include_meta=include_meta
        )
    elif results[0].probs is not None:
        top_id = int(results[0].probs.top1)
        top_score = float(round(float(results[0].probs.top1conf) * 100, 1))
        if top_score >= conf_val * 100:
            detections.append({
                "name": str(model.names[top_id]).title(),
                "confidence": top_score,
                "source": "18-Dish AI"
            })

    annotated_b64 = ""
    if with_plot:
        clean_dets = _strip_meta(detections) if include_meta else detections
        if is_box_model:
            annotated_b64 = draw_top_left_label(image, clean_dets)
        else:
            annotated_b64 = to_base64(Image.fromarray(results[0].plot()[:, :, ::-1]))

    return detections, annotated_b64, is_box_model


def build_dish_prompts(expected_list: list) -> list:
    """Build a strict food-only prompt list (expected dishes or full food dish vocabulary)."""
    prompts = []
    seen = set()

    if expected_list:
        for item in expected_list:
            cp = clean_item_phrase(item)
            if cp and cp not in seen:
                seen.add(cp)
                prompts.append(cp)
            words = cp.split()
            if len(words) >= 2 and words[-1] not in GENERIC_FOOD_WORDS:
                last_w = words[-1]
                if len(last_w) >= 3 and last_w not in seen:
                    seen.add(last_w)
                    prompts.append(last_w)
    else:
        for dish in FOOD_DISH_VOCAB:
            if dish not in seen:
                seen.add(dish)
                prompts.append(dish)

    return prompts


def run_yoloe_dish_prompt(
    image: Image.Image,
    prompt_list: list,
    conf_val: float,
    with_plot: bool = True,
    imgsz: int = 640,
    is_targeted: bool = False,
    keep_meta: bool = False
):
    """Run YOLOE text-prompt model strictly on dish names with class-agnostic NMS & noise filtering."""
    global _cached_prompt_classes
    if yoloe_prompt_model is None or not prompt_list:
        return [], ""

    # Raise floor when scanning 100+ general dish names so low-score color/texture guesses are suppressed
    effective_conf = conf_val if is_targeted else max(conf_val, 0.35)

    prompt_tuple = tuple(prompt_list)
    with _prompt_lock:
        if _cached_prompt_classes != prompt_tuple:
            text_pe = yoloe_prompt_model.get_text_pe(prompt_list)
            yoloe_prompt_model.set_classes(prompt_list, text_pe)
            _cached_prompt_classes = prompt_tuple
        results = yoloe_prompt_model.predict(
            image,
            conf=effective_conf,
            iou=0.45,
            agnostic_nms=True,
            max_det=8,
            imgsz=imgsz,
            verbose=False
        )

    detections = extract_box_detections(
        results[0],
        yoloe_prompt_model.names,
        "Open-Vocab AI",
        img_size=image.size,
        include_meta=True
    )

    # Deduplicate same dish name keeping highest confidence
    unique = {}
    for d in detections:
        k = d["name"].lower()
        if k not in unique or d["confidence"] > unique[k]["confidence"]:
            unique[k] = d
    detections = sorted(unique.values(), key=lambda x: x["confidence"], reverse=True)

    if not is_targeted and detections:
        # Remove tiny low-confidence corner/garnish detections (<4% frame area and <55% confidence)
        filtered = [
            d for d in detections
            if not (d.get("_area", 1.0) < 0.04 and d["confidence"] < 55.0)
        ]
        if filtered:
            detections = filtered

        # If a main dish fills a large portion of the frame, suppress weak secondary hallucinations
        top = detections[0]
        if top.get("_area", 0.0) >= 0.28:
            min_sec = max(42.0, top["confidence"] * 0.65)
            detections = [
                d for i, d in enumerate(detections)
                if i == 0 or d["confidence"] >= min_sec
            ]

    if not keep_meta:
        detections = _strip_meta(detections)

    annotated_b64 = draw_top_left_label(image, detections) if with_plot else ""
    return detections, annotated_b64


def merge_detections(primary: list, secondary: list) -> list:
    merged = {}
    for d in primary + secondary:
        key = d["name"].lower().strip()
        if key not in merged or d["confidence"] > merged[key]["confidence"]:
            merged[key] = d
    out = list(merged.values())
    out.sort(key=lambda x: x["confidence"], reverse=True)
    return out


def run_hybrid_inference(
    image: Image.Image,
    conf_val: float,
    mode: str,
    expected_list: list,
    with_plot: bool = True,
    is_live: bool = False
):
    """Fast dish-only Hybrid pipeline for Live Scan, Capture, and Upload."""
    mode = (mode or "hybrid").strip().lower()
    imgsz = 320 if is_live else 640

    # 1. Pure 18-Dish Custom Model
    if mode == "custom" or yoloe_prompt_model is None:
        dets, ann, _ = run_yolo_on_image(image, conf_val, with_plot=with_plot, imgsz=imgsz)
        return dets, ann

    # 2. Pure Open-Vocab Any Dish Mode (strictly dish vocabulary)
    if mode == "open":
        prompts = build_dish_prompts(expected_list)
        dets, ann = run_yoloe_dish_prompt(
            image, prompts, conf_val, with_plot=with_plot, imgsz=imgsz, is_targeted=bool(expected_list)
        )
        if expected_list and not dets and not is_live:
            dets, ann = run_yoloe_dish_prompt(
                image, build_dish_prompts([]), conf_val, with_plot=with_plot, imgsz=imgsz, is_targeted=False
            )
        return dets, ann

    # 3. Auto Hybrid Mode with Expected Order
    if expected_list:
        trained_exp = [e for e in expected_list if matches_trained_menu(e) is not None]
        open_exp = [e for e in expected_list if matches_trained_menu(e) is None]

        if trained_exp and not open_exp:
            dets, ann, _ = run_yolo_on_image(image, conf_val, with_plot=with_plot, imgsz=imgsz)
            if dets or is_live:
                return dets, ann
            return run_yoloe_dish_prompt(
                image, build_dish_prompts(trained_exp), conf_val, with_plot=with_plot, imgsz=imgsz, is_targeted=True
            )

        if open_exp and not trained_exp:
            prompts = build_dish_prompts(open_exp)
            dets, ann = run_yoloe_dish_prompt(
                image, prompts, conf_val, with_plot=with_plot, imgsz=imgsz, is_targeted=True
            )
            if not dets and not is_live:
                dets, ann = run_yoloe_dish_prompt(
                    image, build_dish_prompts([]), conf_val, with_plot=with_plot, imgsz=imgsz, is_targeted=False
                )
            return dets, ann

        custom_dets, _, _ = run_yolo_on_image(image, conf_val, with_plot=False, imgsz=imgsz)
        open_dets, _ = run_yoloe_dish_prompt(
            image, build_dish_prompts(open_exp), conf_val, with_plot=False, imgsz=imgsz, is_targeted=True
        )
        dets = merge_detections(custom_dets, open_dets)
        ann = draw_top_left_label(image, dets) if with_plot else ""
        return dets, ann

    # 4. Detection-Only (No Expected Order — Reference Photo Upload, Meal Upload, or Live Scan)
    custom_dets, _, is_box_model = run_yolo_on_image(
        image, conf_val, with_plot=False, imgsz=imgsz, include_meta=True
    )

    if is_live and custom_dets and custom_dets[0]["confidence"] >= 65.0:
        return _strip_meta(custom_dets), ""

    open_dets, _ = run_yoloe_dish_prompt(
        image,
        build_dish_prompts([]),
        conf_val,
        with_plot=False,
        imgsz=imgsz,
        is_targeted=False,
        keep_meta=True
    )

    if is_box_model:
        kept_open = []
        for od in open_dets:
            if od["confidence"] < 55.0:
                continue
            od_box = od.get("_bbox")
            overlaps_custom = (
                od_box is not None and any(
                    cd.get("_bbox") is not None and _box_iou(od_box, cd["_bbox"]) >= 0.35
                    for cd in custom_dets
                )
            )
            if not overlaps_custom:
                kept_open.append(od)
        dets = _strip_meta(merge_detections(custom_dets, kept_open))
        ann = draw_top_left_label(image, dets) if with_plot else ""
        return dets, ann

    # Custom model is a classifier (probs): protect confident 18-dish predictions from weak open-vocab noise
    custom_top_conf = custom_dets[0]["confidence"] if custom_dets else 0.0
    trained_low = {n.lower() for n in get_trained_class_names()}

    if custom_top_conf >= 65.0:
        strong_non_18 = [
            d for d in open_dets
            if d["name"].lower() not in trained_low
            and d["confidence"] >= max(68.0, custom_top_conf)
        ]
        if strong_non_18:
            dets = _strip_meta(strong_non_18)
            ann = draw_top_left_label(image, dets) if with_plot else ""
            return dets, ann

        distinct_sides = [
            d for d in open_dets
            if d["name"].lower() != custom_dets[0]["name"].lower()
            and d["confidence"] >= 62.0
        ]
        dets = _strip_meta(merge_detections(custom_dets, distinct_sides))
        ann = draw_top_left_label(image, dets) if with_plot else ""
        return dets, ann

    if open_dets:
        dets = _strip_meta(open_dets)
        ann = draw_top_left_label(image, dets) if with_plot else ""
        return dets, ann

    dets = _strip_meta(custom_dets)
    ann = draw_top_left_label(image, dets) if with_plot else ""
    return dets, ann


@app.get("/")
def home():
    return FileResponse(
        os.path.join(BASE_DIR, "static", "index.html"),
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate"
        }
    )


@app.get("/logo")
def get_logo():
    for folder in ["static", "."]:
        for name in [
            "logo.jpeg", "logo.jpg", "logo.png", "logo.webp"
        ]:
            path = os.path.join(BASE_DIR, folder, name)

            if os.path.exists(path):
                return FileResponse(path)

    raise HTTPException(
        status_code=404,
        detail="Logo not found"
    )


@app.get("/classes")
def classes():
    if model is not None:
        names = sorted([
            str(n).title() for n in model.names.values()
        ])

        return {
            "classes": names,
            "total": int(len(names)),
            "demo": False,
            "hybrid": bool(yoloe_prompt_model is not None)
        }

    return {
        "classes": DEMO_CLASSES,
        "total": int(len(DEMO_CLASSES)),
        "demo": True,
        "hybrid": bool(yoloe_prompt_model is not None)
    }


@app.get("/orders")
def list_orders(q: str = ""):
    """Return persisted dispatch orders from SQLite."""

    query = q.strip().lower()

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row

        if query:
            like = f"%{query}%"

            rows = conn.execute(
                """
                SELECT * FROM dispatch_orders
                WHERE LOWER(order_id) LIKE ?
                OR LOWER(detected_json) LIKE ?
                OR LOWER(expected_json) LIKE ?
                ORDER BY id DESC LIMIT 100
                """,
                (like, like, like)
            ).fetchall()

        else:
            rows = conn.execute(
                """
                SELECT * FROM dispatch_orders
                ORDER BY id DESC LIMIT 100
                """
            ).fetchall()

    results = []

    for r in rows:
        results.append({
            "dbId": r["id"],
            "orderId": r["order_id"],
            "timestamp": r["timestamp"],
            "time": r["display_time"],
            "expected": json.loads(r["expected_json"]),
            "detected": json.loads(r["detected_json"]),
            "fill_pct": r["fill_pct"],
            "min_fill": r["min_fill"],
            "missing": json.loads(r["missing_json"]),
            "status": r["status"],
            "proofHash": r["proof_hash"],
            "raw": json.loads(r["payload_json"])
        })

    return {"orders": results}


@app.post("/orders")
def save_order(payload: dict = Body(...)):
    """Persist a verified order permanently in SQLite."""

    order_id = str(payload.get("orderId", "ORD-101"))
    display_time = str(
        payload.get("time", datetime.now().strftime("%I:%M %p"))
    )

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    expected = payload.get("expected", [])
    detected = payload.get("detected", [])
    fill_pct = float(payload.get("fill_pct", 0.0))
    min_fill = float(payload.get("min_fill", 85.0))
    missing = payload.get("missing", [])
    status = str(payload.get("status", "DETECTION_ONLY"))
    raw = payload.get("raw", {})

    img_b64 = (
        raw.get("annotated_image", "")
        or raw.get("original_image", "")
    )

    proof_hash = compute_proof_hash(
        order_id,
        timestamp,
        fill_pct,
        detected,
        status,
        img_b64
    )

    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            """
            INSERT INTO dispatch_orders
            (
                order_id, timestamp, display_time,
                expected_json, detected_json,
                fill_pct, min_fill, missing_json,
                status, proof_hash, payload_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                order_id,
                timestamp,
                display_time,
                json.dumps(expected),
                json.dumps(detected),
                fill_pct,
                min_fill,
                json.dumps(missing),
                status,
                proof_hash,
                json.dumps(raw)
            )
        )

        conn.commit()
        db_id = cur.lastrowid

    return {
        "dbId": db_id,
        "timestamp": timestamp,
        "proofHash": proof_hash
    }


@app.patch("/orders/{db_id}")
def update_order_status(db_id: int, payload: dict = Body(...)):
    """Update order status on Human Verification."""

    new_status = str(payload.get("status", "")).strip()

    if not new_status:
        raise HTTPException(
            status_code=400,
            detail="Missing status"
        )

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row

        row = conn.execute(
            "SELECT * FROM dispatch_orders WHERE id = ?",
            (db_id,)
        ).fetchone()

        if not row:
            raise HTTPException(
                status_code=404,
                detail="Order not found"
            )

        raw = json.loads(row["payload_json"])

        if "verification" in raw:
            raw["verification"]["status"] = new_status

        detected = json.loads(row["detected_json"])

        img_b64 = (
            raw.get("annotated_image", "")
            or raw.get("original_image", "")
        )

        new_hash = compute_proof_hash(
            row["order_id"],
            row["timestamp"],
            row["fill_pct"],
            detected,
            new_status,
            img_b64
        )

        conn.execute(
            """
            UPDATE dispatch_orders
            SET status = ?, proof_hash = ?, payload_json = ?
            WHERE id = ?
            """,
            (
                new_status,
                new_hash,
                json.dumps(raw),
                db_id
            )
        )

        conn.commit()

    return {
        "dbId": db_id,
        "status": new_status,
        "proofHash": new_hash
    }


@app.post("/fetch-csv")
async def fetch_csv_urls(file: UploadFile = File(...)):
    """Parse an uploaded CSV and persist it in SQLite until manually closed."""

    try:
        raw_bytes = await file.read()
        text = raw_bytes.decode(
            "utf-8-sig",
            errors="ignore"
        )

    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Could not read CSV file."
        )

    reader = csv.reader(io.StringIO(text))

    rows = [
        r for r in reader
        if any(cell.strip() for cell in r)
    ]

    if not rows:
        raise HTTPException(
            status_code=400,
            detail="CSV file is empty."
        )

    headers = [h.strip().lower() for h in rows[0]]

    has_header = any(
        h in (
            "image_url", "url", "link",
            "img", "image", "item_name", "item_id"
        )
        for h in headers
    )

    url_idx = None
    name_idx = None
    id_idx = None

    if has_header:
        for col in ("image_url", "url", "link", "img", "image"):
            if col in headers:
                url_idx = headers.index(col)
                break

        for col in ("item_name", "name", "dish", "title"):
            if col in headers:
                name_idx = headers.index(col)
                break

        if "item_id" in headers:
            id_idx = headers.index("item_id")

        data_rows = rows[1:]

    else:
        data_rows = rows

    items = []
    seen_urls = set()

    for r in data_rows:
        img_url = ""

        if url_idx is not None and url_idx < len(r):
            candidate = r[url_idx].strip()

            if candidate.startswith(("http://", "https://")):
                img_url = candidate

        if not img_url:
            for cell in r:
                c = cell.strip()

                if c.startswith(("http://", "https://")):
                    img_url = c
                    break

        if not img_url or img_url in seen_urls:
            continue

        seen_urls.add(img_url)

        item_name = ""

        if name_idx is not None and name_idx < len(r):
            item_name = r[name_idx].strip().replace("\xa0", " ")

        elif len(r) >= 2 and url_idx != 0:
            item_name = r[0].strip()

        item_id = (
            r[id_idx].strip()
            if id_idx is not None and id_idx < len(r)
            else ""
        )

        items.append({
            "id": item_id,
            "name": item_name or f"Item {len(items) + 1}",
            "url": img_url
        })

    if not items:
        raise HTTPException(
            status_code=400,
            detail="No valid image URLs found in CSV."
        )

    filename = file.filename or "menu.csv"

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO csv_store (id, filename, items_json)
            VALUES (1, ?, ?)
            """,
            (filename, json.dumps(items))
        )
        conn.commit()

    return {
        "filename": filename,
        "items": items,
        "total": len(items)
    }


@app.get("/csv-items")
def get_saved_csv():
    """Return persisted CSV gallery items if a CSV is stored."""

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT filename, items_json FROM csv_store WHERE id = 1"
        ).fetchone()

    if not row:
        return {"filename": "", "items": [], "total": 0}

    items = json.loads(row["items_json"])
    return {
        "filename": row["filename"],
        "items": items,
        "total": len(items)
    }


@app.delete("/csv-items")
def clear_saved_csv():
    """Manually close/remove the persisted CSV file from SQLite."""

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM csv_store WHERE id = 1")
        conn.commit()

    return {"cleared": True}


@app.get("/proxy-image")
def proxy_image(url: str):
    """Fetch an external image URL via backend."""

    clean_url = url.strip()

    if not clean_url.startswith(("http://", "https://")):
        raise HTTPException(
            status_code=400,
            detail="Invalid URL"
        )

    try:
        req = urllib.request.Request(
            clean_url,
            headers={"User-Agent": "Mozilla/5.0"}
        )

        with urllib.request.urlopen(req, timeout=10) as resp:
            content_type = (
                resp.headers.get("Content-Type", "image/jpeg")
                .split(";")[0].strip()
            )
            data = resp.read()

        if not content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail="Not an image"
            )

        return Response(
            content=data,
            media_type=content_type,
            headers={"Cache-Control": "public, max-age=3600"}
        )

    except HTTPException:
        raise

    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Could not fetch image"
        )


@app.post("/predict-url")
async def predict_from_url(
    url: str = Form(...),
    conf: float = Form(0.25)
):
    """Fetch a reference image from a URL or extract images from a page."""

    clean_url = url.strip()

    if not clean_url.startswith(("http://", "https://")):
        raise HTTPException(
            status_code=400,
            detail="URL must start with http:// or https://"
        )

    try:
        req = urllib.request.Request(
            clean_url,
            headers={"User-Agent": "Mozilla/5.0"}
        )

        with urllib.request.urlopen(req, timeout=10) as resp:
            content_type = resp.headers.get(
                "Content-Type", ""
            ).lower()
            raw_bytes = resp.read()

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Could not fetch URL: {str(e)}"
        )

    if "text/html" in content_type:
        html = raw_bytes.decode("utf-8", errors="ignore")

        raw_srcs = re.findall(
            r'<img[^>]+src=["\']([^"\']+)["\']',
            html,
            flags=re.IGNORECASE
        )

        images = []
        seen = set()

        for src in raw_srcs:
            abs_url = urljoin(clean_url, src)
            low = abs_url.lower()

            if (
                abs_url in seen
                or low.endswith(".svg")
                or "favicon" in low
            ):
                continue

            seen.add(abs_url)
            images.append(abs_url)

        return {
            "type": "gallery",
            "images": images[:20]
        }

    try:
        image = ImageOps.exif_transpose(
            Image.open(io.BytesIO(raw_bytes))
        ).convert("RGB")

    except Exception:
        raise HTTPException(
            status_code=400,
            detail="URL did not return a valid image."
        )

    fill_pct = float(estimate_fill_percentage(image))

    detections, _ = run_hybrid_inference(
        image,
        float(conf),
        mode="hybrid",
        expected_list=[],
        with_plot=False,
        is_live=False
    )

    return {
        "type": "image",
        "image_b64": to_base64(image),
        "detections": detections,
        "fill_pct": fill_pct
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    conf: float = Form(0.25),
    expected: str = Form(""),
    min_fill: float = Form(85.0),
    live: bool = Form(False),
    mode: str = Form("hybrid")
):
    start = time.perf_counter()

    try:
        img_bytes = await file.read()

        image = ImageOps.exif_transpose(
            Image.open(io.BytesIO(img_bytes))
        ).convert("RGB")

    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid image file."
        )

    fill_pct = float(estimate_fill_percentage(image))
    min_fill_val = float(min_fill)

    quantity_ok = bool(fill_pct >= min_fill_val)
    conf_val = float(conf)

    expected_list = [
        item.strip()
        for item in expected.split(",")
        if item.strip()
    ]

    original_b64 = "" if live else to_base64(image)

    detections, annotated_b64 = run_hybrid_inference(
        image,
        conf_val,
        mode=mode,
        expected_list=expected_list,
        with_plot=(not live),
        is_live=live
    )

    speed_ms = float(
        round((time.perf_counter() - start) * 1000, 1)
    )

    detected_names = {
        d["name"] for d in detections
    }

    matched = [
        item for item in expected_list
        if is_expected_matched(item, detected_names)
    ]

    missing = [
        item for item in expected_list
        if item not in matched
    ]

    extra = (
        [
            name for name in detected_names
            if not any(is_expected_matched(exp, {name}) for exp in expected_list)
        ]
        if expected_list else []
    )

    if not quantity_ok:
        status = "DEFECT_QUANTITY"

    elif not expected_list:
        status = "DETECTION_ONLY"

    elif len(missing) == 0:
        status = "VERIFIED"

    else:
        status = "INCOMPLETE"

    return {
        "original_image": original_b64,
        "annotated_image": annotated_b64,
        "detections": detections,
        "count": int(len(detections)),
        "speed_ms": speed_ms,

        "quantity": {
            "fill_pct": float(fill_pct),
            "min_fill": float(min_fill_val),
            "ok": bool(quantity_ok)
        },

        "verification": {
            "status": status,
            "expected": expected_list,
            "matched": matched,
            "missing": missing,
            "extra": extra
        }
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )