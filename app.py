import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TORCH_NUM_THREADS"] = "1"
os.environ["YOLO_VERBOSE"] = "False"

import gc
import io, base64, time, re, json, hashlib, csv, threading
import urllib.request, urllib.error
from urllib.parse import urljoin
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from contextlib import closing

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Body
from fastapi.responses import FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps
import cv2
cv2.setNumThreads(0)
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor

app = FastAPI(title="Mother's Hut Meal Verification System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

_env_path = os.path.join(BASE_DIR, ".env")
if os.path.exists(_env_path):
    with open(_env_path, "r", encoding="utf-8") as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))

ONNX_PATH = os.path.join(BASE_DIR, "best.onnx")
META_PATH = os.path.join(BASE_DIR, "model_meta.json")
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres.sldwvllytyhnnebitvzg:Arghya19012006@aws-1-ap-northeast-2.pooler.supabase.com:6543/postgres"
)
METRICS_DATABASE_URL = os.environ.get(
    "METRICS_DATABASE_URL",
    "postgresql://postgres.vmdxwszdawcdluaignpl:The_Neotia_Uni@aws-1-ap-southeast-1.pooler.supabase.com:6543/postgres"
)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite").strip()

IST = ZoneInfo("Asia/Kolkata")


def get_current_ist_datetime() -> datetime:
    """Return current timestamp strictly in Indian Standard Time (IST)."""
    return datetime.now(IST)

_gemini_cache = {}
_gemini_cache_order = []
_gemini_lock = threading.Lock()
_gemini_cooldown_until = 0.0


def get_db():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL environment variable is not set.")
    return psycopg2.connect(DATABASE_URL, sslmode="require", connect_timeout=10)


def get_metrics_db():
    if not METRICS_DATABASE_URL:
        raise RuntimeError("METRICS_DATABASE_URL is not set.")
    return psycopg2.connect(METRICS_DATABASE_URL, sslmode="require", connect_timeout=10)


def update_food_quality_metric(dt: datetime = None, current_score: float = None):
    """
    Update food_quality_score in master_metrics on the metrics database in 5-minute intervals.
    The window_start aligns on minutes ending in 0 or 5 (e.g. 18:00, 18:05, 18:10) in IST.
    If the window exists, updates food_quality_score with the exact average score.
    If not, creates the record with the interval.
    """
    if not METRICS_DATABASE_URL:
        return
    try:
        if dt is not None:
            now = dt.astimezone(IST) if dt.tzinfo else dt.replace(tzinfo=IST)
        else:
            now = get_current_ist_datetime()

        # Format naive datetime for consistent timestamp comparison matching the IST clock
        now_naive = now.replace(tzinfo=None)

        # Floor minute to multiple of 5 (e.g. 18:00, 18:05, 18:10, 18:15, 18:20)
        floored_minute = (now_naive.minute // 5) * 5
        window_start = now_naive.replace(minute=floored_minute, second=0, microsecond=0)

        # Compute average score for the current 5-minute interval across all orders tested
        scores = []
        if current_score is not None:
            scores.append(float(current_score))

        try:
            with closing(get_db()) as conn_disp:
                with conn_disp.cursor() as cur_disp:
                    window_start_str = window_start.strftime("%Y-%m-%d %H:%M:%S")
                    window_end_str = (window_start + timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S")
                    cur_disp.execute(
                        """
                        SELECT expected_json, detected_json, payload_json
                        FROM dispatch_orders
                        WHERE timestamp >= %s AND timestamp < %s
                        """,
                        (window_start_str, window_end_str)
                    )
                    rows = cur_disp.fetchall()
                    for r in rows:
                        try:
                            exp = json.loads(r[0]) if r[0] else []
                            pay = json.loads(r[2]) if r[2] else {}
                            dets = pay.get("detections", [])
                            if not dets and r[1]:
                                dets = [{"name": name, "confidence": 85.0} for name in json.loads(r[1])]
                            c_score = calculate_match_confidence(exp, dets)
                            if c_score > 0:
                                scores.append(c_score)
                        except Exception:
                            continue
        except Exception as e:
            print(f"⚠️ Notice: could not fetch interval orders: {e}")

        avg_score = float(round(sum(scores) / len(scores), 2)) if scores else 0.0

        with closing(get_metrics_db()) as conn, conn:
            with conn.cursor() as cur:
                # Check if window_start already exists in master_metrics
                cur.execute(
                    "SELECT food_quality_score FROM master_metrics WHERE window_start = %s",
                    (window_start,)
                )
                existing = cur.fetchone()

                if existing:
                    # Update exact value replacing 0 or prior value
                    cur.execute(
                        """
                        UPDATE master_metrics
                        SET food_quality_score = %s
                        WHERE window_start = %s
                        """,
                        (avg_score, window_start)
                    )
                else:
                    # Insert new window_start with proper 5-min interval
                    cur.execute(
                        """
                        INSERT INTO master_metrics
                        (window_start, vehicle_count, total_orders, hygiene_score, food_quality_score, positive_sentiment, negative_sentiment, customer_count)
                        VALUES (%s, 0, 0, 0, %s, 0, 0, 0)
                        """,
                        (window_start, avg_score)
                    )
    except Exception as e:
        print(f"⚠️ Failed to update food_quality_score in master_metrics: {e}")


model = None
ort_input_name = None
model_task = "classify"
model_imgsz = 224
model_names = {}
yoloe_prompt_model = None
_cached_prompt_classes = None
_prompt_lock = threading.Lock()

if os.path.exists(ONNX_PATH) and os.path.exists(META_PATH):
    import onnxruntime as ort
    with open(META_PATH, "r") as f:
        _meta = json.load(f)
    model_task = _meta.get("task", "classify")
    model_imgsz = int(_meta.get("imgsz", 224 if model_task == "classify" else 640))
    model_names = {int(k): v for k, v in _meta.get("names", {}).items()}

    _sess_opts = ort.SessionOptions()
    _sess_opts.intra_op_num_threads = 1
    _sess_opts.inter_op_num_threads = 1
    model = ort.InferenceSession(
        ONNX_PATH, sess_options=_sess_opts, providers=["CPUExecutionProvider"]
    )
    ort_input_name = model.get_inputs()[0].name
    print(f"✅ Loaded ONNX model '{ONNX_PATH}' ({model_task}, imgsz={model_imgsz}, {len(model_names)} classes)")
else:
    print("⚠️ best.onnx not found — running custom model in Demo Mode")

DEMO_CLASSES = [
    "Aloo Gobi", "Biryani", "Butter Chicken", "Chana Masala",
    "Chapati", "Dal Makhani", "Gulab Jamun", "Jalebi", "Naan", "Palak Paneer"
]

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
    if not DATABASE_URL:
        print("⚠️ DATABASE_URL not set — skipping DB init until configured")
        return
    with closing(get_db()) as conn, conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS dispatch_orders (
                    id SERIAL PRIMARY KEY,
                    order_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    display_time TEXT NOT NULL,
                    expected_json TEXT NOT NULL,
                    detected_json TEXT NOT NULL,
                    fill_pct DOUBLE PRECISION NOT NULL,
                    min_fill DOUBLE PRECISION NOT NULL,
                    missing_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    proof_hash TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS csv_store (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    filename TEXT NOT NULL,
                    items_json TEXT NOT NULL
                )
            """)


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


try:
    init_db()
except Exception as e:
    print(f"⚠️ Initial database connection postponed: {e}")


def to_base64(pil_img: Image.Image) -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=88)
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
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
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


def estimate_fill_percentage(pil_img: Image.Image, detections: list | None = None) -> float:
    """
    Estimate bowl/plate food fill percentage (0-100%) using color and texture.
    Includes human skin suppression so people or hands in front of the camera do not register as full meal bowls.
    """
    if detections is not None and len(detections) == 0:
        return 0.0

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

    h_ch = hsv[:, :, 0]
    s_ch = hsv[:, :, 1]
    v_ch = hsv[:, :, 2]

    # Skin color suppression in HSV (human skin tones: hands, face)
    skin_mask = (((h_ch <= 25) | (h_ch >= 165)) & (s_ch >= 18) & (s_ch <= 160) & (v_ch >= 50))
    smooth_skin = skin_mask & (local_std < 22)

    rich_color = (s_ch >= 42) & (v_ch >= 30) & (v_ch <= 245) & (~smooth_skin)
    textured_food = (
        (local_std >= 18) & (s_ch >= 22) & (v_ch >= 35) & (v_ch <= 240) & (~smooth_skin)
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
        return [str(n).title() for n in model_names.values()]
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


def calculate_match_confidence(expected_items: list, detections: list) -> float:
    """
    Calculate confidence percentage that the expected items match the uploaded items.
    If expected items are provided:
      Computes the match confidence for each expected item based on the detected item confidence.
    If no expected items provided:
      Uses the average confidence of detected items (or top confidence).
    Returns confidence score as a percentage (0.0 to 100.0).
    """
    if not detections:
        return 0.0

    if expected_items:
        confidences = []
        for exp in expected_items:
            best_conf = 0.0
            for d in detections:
                dname = d.get("name", "")
                conf = float(d.get("confidence", 0.0))
                if is_expected_matched(exp, {dname}):
                    if conf > best_conf:
                        best_conf = conf
            if best_conf > 0.0:
                confidences.append(best_conf)
        if confidences:
            return float(round(sum(confidences) / len(confidences), 2))
        return 0.0
    else:
        confs = [float(d.get("confidence", 0.0)) for d in detections if d.get("confidence") is not None]
        return float(round(sum(confs) / len(confs), 2)) if confs else 0.0


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


def _preprocess_classify(pil_img: Image.Image, size: int) -> np.ndarray:
    w, h = pil_img.size
    scale = size / min(w, h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = pil_img.resize((nw, nh), Image.BILINEAR)
    left = (nw - size) // 2
    top = (nh - size) // 2
    cropped = resized.crop((left, top, left + size, top + size))
    arr = np.asarray(cropped, dtype=np.float32) / 255.0
    return np.expand_dims(np.transpose(arr, (2, 0, 1)), axis=0)


def _preprocess_classify_full(pil_img: Image.Image, size: int) -> np.ndarray:
    resized = pil_img.resize((size, size), Image.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    return np.expand_dims(np.transpose(arr, (2, 0, 1)), axis=0)


def _to_probs(out: np.ndarray) -> np.ndarray:
    if out.min() < 0 or abs(float(out.sum()) - 1.0) > 0.05:
        exp = np.exp(out - np.max(out))
        return exp / exp.sum()
    return out


def is_valid_food_prediction(probs: np.ndarray, top_id: int, conf_val: float) -> tuple[bool, float]:
    """
    Guard against non-food inputs (hands, empty plates, background objects).
    A true dish prediction in an 18-class softmax model exhibits a distinct peak.
    Non-food inputs produce flat distributions with high entropy and small margins.
    """
    top_score = float(round(float(probs[top_id]) * 100, 1))
    
    # Absolute minimum confidence required (must meet user threshold and at least 35%)
    min_required = max(conf_val * 100, 35.0)
    if top_score < min_required:
        return False, top_score

    # Check margin over runner-up class
    sorted_probs = np.sort(probs)
    runner_up = float(sorted_probs[-2]) if len(sorted_probs) > 1 else 0.0
    margin = float(probs[top_id]) - runner_up

    # If top score is between 35% and 55%, require a clear margin (at least 8%)
    # to avoid false positives on background objects/hands
    if top_score < 55.0 and margin < 0.08:
        return False, top_score

    return True, top_score


def run_yolo_on_image(
    image: Image.Image,
    conf_val: float,
    with_plot: bool = True,
    imgsz: int = 640,
    include_meta: bool = False,
    expected_list: list | None = None,
    is_live: bool = False
):
    """Run custom 18-dish YOLO ONNX model with non-food suppression."""
    if model is None:
        dets = [{"name": "Butter Chicken", "confidence": 91.7, "source": "18-Dish AI"}]
        ann = draw_top_left_label(image, dets) if with_plot else ""
        return dets, ann, False

    detections = []
    inp = _preprocess_classify(image, model_imgsz)
    out = model.run(None, {ort_input_name: inp})[0][0]
    probs = _to_probs(out)

    if not is_live:
        inp_full = _preprocess_classify_full(image, model_imgsz)
        out_full = model.run(None, {ort_input_name: inp_full})[0][0]
        probs = (0.65 * probs) + (0.35 * _to_probs(out_full))
        del inp_full, out_full

    top_id = int(np.argmax(probs))
    is_valid, top_score = is_valid_food_prediction(probs, top_id, conf_val)

    if is_valid:
        detections.append({
            "name": str(model_names.get(top_id, top_id)).title(),
            "confidence": top_score,
            "source": "18-Dish AI"
        })

    # Only look for secondary expected items if the primary prediction is valid food
    if is_valid and expected_list and len(expected_list) >= 2:
        for cid in np.argsort(probs)[::-1][:3]:
            cid = int(cid)
            if cid == top_id:
                continue
            sc = float(round(float(probs[cid]) * 100, 1))
            cname = str(model_names.get(cid, cid)).title()
            if sc >= max(conf_val * 100, 25.0) and any(
                is_expected_matched(exp, {cname}) for exp in expected_list
            ):
                detections.append({
                    "name": cname,
                    "confidence": sc,
                    "source": "18-Dish AI"
                })

    del inp, out
    gc.collect()

    annotated_b64 = draw_top_left_label(image, detections) if with_plot else ""
    return detections, annotated_b64, False


def run_gemini_vision(
    image: Image.Image,
    conf_val: float,
    expected_list: list | None = None
) -> list:
    """Zero-RAM open-vocab dish detection via Gemini Flash REST API (with cache & 429 cooldown)."""
    global _gemini_cooldown_until
    if not GEMINI_API_KEY or time.time() < _gemini_cooldown_until:
        return []

    gemini_img = image.copy()
    gemini_img.thumbnail((512, 512))
    buf = io.BytesIO()
    gemini_img.save(buf, format="JPEG", quality=70)
    raw_bytes = buf.getvalue()
    del gemini_img

    cache_key = (
        hashlib.md5(raw_bytes).hexdigest(),
        tuple(sorted(e.lower() for e in (expected_list or [])))
    )

    with _gemini_lock:
        if cache_key in _gemini_cache:
            cached = _gemini_cache[cache_key]
            return [d for d in cached if d["confidence"] >= conf_val * 100]

    img_b64 = base64.b64encode(raw_bytes).decode("utf-8")
    del raw_bytes

    hint_items = expected_list if expected_list else FOOD_DISH_VOCAB
    prompt = (
        "You are an Indian & restaurant meal QA classifier. "
        "Identify every distinct prepared dish visible on the plate/bowl. "
        f"Prefer matching exact dish names from this list when applicable: {', '.join(hint_items[:120])}. "
        'Return ONLY valid JSON in this exact format: '
        '{"dishes": [{"name": "Dish Name", "confidence": 92}]}. '
        'Confidence must be an integer 1-99. If no food is visible, return {"dishes": []}.'
    )

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    )
    gen_cfg = {"responseMimeType": "application/json"}
    if "lite" in GEMINI_MODEL.lower():
        gen_cfg["thinkingConfig"] = {"thinkingLevel": "minimal"}

    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": "image/jpeg", "data": img_b64}}
            ]
        }],
        "generationConfig": gen_cfg
    }

    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=25) as resp:
            body = json.loads(resp.read().decode("utf-8"))

        text = body["candidates"][0]["content"]["parts"][0]["text"]
        parsed = json.loads(text)
        raw_out = []
        for item in parsed.get("dishes", []):
            name = str(item.get("name", "")).strip().title()
            score = float(min(99.0, max(1.0, float(item.get("confidence", 85)))))
            if name:
                raw_out.append({
                    "name": name,
                    "confidence": round(score, 1),
                    "source": "Open-Vocab AI"
                })
        raw_out.sort(key=lambda x: x["confidence"], reverse=True)

        with _gemini_lock:
            _gemini_cache[cache_key] = raw_out
            _gemini_cache_order.append(cache_key)
            if len(_gemini_cache_order) > 32:
                old_key = _gemini_cache_order.pop(0)
                _gemini_cache.pop(old_key, None)

        return [d for d in raw_out if d["confidence"] >= conf_val * 100]
    except urllib.error.HTTPError as e:
        if e.code == 429:
            _gemini_cooldown_until = time.time() + 60.0
            print("⚠️ Gemini rate limit (429) — cooling down 60s, using best.onnx")
        else:
            print(f"⚠️ Gemini HTTP {e.code} fallback — using best.onnx")
        return []
    except Exception as e:
        print(f"⚠️ Gemini fallback ({e}) — using best.onnx")
        return []


def run_hybrid_inference(
    image: Image.Image,
    conf_val: float,
    mode: str,
    expected_list: list,
    with_plot: bool = True,
    is_live: bool = False
):
    """Hybrid ONNX (18-Dish) + rate-limited zero-RAM Gemini Open-Vocab pipeline."""
    mode = (mode or "hybrid").strip().lower()
    imgsz = 320 if is_live else 640

    if is_live or mode == "custom" or not GEMINI_API_KEY:
        dets, ann, _ = run_yolo_on_image(
            image, conf_val, with_plot=with_plot, imgsz=imgsz,
            expected_list=expected_list, is_live=is_live
        )
        return dets, ann

    if mode == "open":
        dets = run_gemini_vision(image, conf_val, expected_list=expected_list)
        if dets:
            ann = draw_top_left_label(image, dets) if with_plot else ""
            return dets, ann
        dets, ann, _ = run_yolo_on_image(
            image, conf_val, with_plot=with_plot, imgsz=imgsz,
            expected_list=expected_list, is_live=False
        )
        return dets, ann

    custom_dets, _, _ = run_yolo_on_image(
        image, conf_val, with_plot=False, imgsz=imgsz,
        expected_list=expected_list, is_live=False
    )
    open_exp = [e for e in expected_list if matches_trained_menu(e) is None]
    top_conf = custom_dets[0]["confidence"] if custom_dets else 0.0
    det_names = {d["name"] for d in custom_dets}
    unmatched_18 = bool(expected_list) and not open_exp and not any(
        is_expected_matched(exp, det_names) for exp in expected_list
    )

    # Only call Gemini if non-18 item expected, 18-dish missed/low confidence, or uncertain
    need_open_vocab = (
        bool(open_exp)
        or unmatched_18
        or (not expected_list and top_conf < 82.0)
        or (top_conf < 70.0)
    )
    if need_open_vocab:
        open_dets = run_gemini_vision(image, conf_val, expected_list=expected_list)
        if open_dets:
            if open_exp or unmatched_18 or top_conf < 80.0:
                dets = open_dets
            else:
                trained_low = {n.lower() for n in get_trained_class_names()}
                extras = [d for d in open_dets if d["name"].lower() not in trained_low]
                dets = custom_dets + extras
            ann = draw_top_left_label(image, dets) if with_plot else ""
            return dets, ann

    ann = draw_top_left_label(image, custom_dets) if with_plot else ""
    return custom_dets, ann


@app.get("/")
def home():
    return FileResponse(
        os.path.join(BASE_DIR, "static", "index.html"),
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"}
    )


@app.get("/logo")
def get_logo():
    for folder in ["static", "."]:
        for name in ["logo.jpeg", "logo.jpg", "logo.png", "logo.webp"]:
            path = os.path.join(BASE_DIR, folder, name)
            if os.path.exists(path):
                return FileResponse(path)
    raise HTTPException(status_code=404, detail="Logo not found")


@app.get("/classes")
def classes():
    if model is not None:
        names = sorted([str(n).title() for n in model_names.values()])
        return {
            "classes": names,
            "total": int(len(names)),
            "demo": False,
            "hybrid": bool(GEMINI_API_KEY)
        }

    return {
        "classes": DEMO_CLASSES,
        "total": int(len(DEMO_CLASSES)),
        "demo": True,
        "hybrid": bool(GEMINI_API_KEY)
    }


@app.get("/orders")
def list_orders(q: str = ""):
    """Return persisted dispatch orders from PostgreSQL."""
    query = q.strip().lower()

    with closing(get_db()) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            if query:
                like = f"%{query}%"
                cur.execute(
                    """
                    SELECT * FROM dispatch_orders
                    WHERE LOWER(order_id) LIKE %s
                    OR LOWER(detected_json) LIKE %s
                    OR LOWER(expected_json) LIKE %s
                    ORDER BY id DESC LIMIT 100
                    """,
                    (like, like, like)
                )
            else:
                cur.execute("SELECT * FROM dispatch_orders ORDER BY id DESC LIMIT 100")
            rows = cur.fetchall()

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
    """Persist a verified order permanently in PostgreSQL."""
    now_ist = get_current_ist_datetime()
    order_id = str(payload.get("orderId", "ORD-101"))
    display_time = str(payload.get("time", "")).strip() or now_ist.strftime("%H:%M")
    timestamp = now_ist.strftime("%Y-%m-%d %H:%M:%S")

    expected = payload.get("expected", [])
    detected = payload.get("detected", [])
    fill_pct = float(payload.get("fill_pct", 0.0))
    min_fill = float(payload.get("min_fill", 85.0))
    missing = payload.get("missing", [])
    status = str(payload.get("status", "DETECTION_ONLY"))
    raw = payload.get("raw", {})

    img_b64 = raw.get("annotated_image", "") or raw.get("original_image", "")

    proof_hash = compute_proof_hash(
        order_id, timestamp, fill_pct, detected, status, img_b64
    )

    with closing(get_db()) as conn, conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO dispatch_orders
                (
                    order_id, timestamp, display_time,
                    expected_json, detected_json,
                    fill_pct, min_fill, missing_json,
                    status, proof_hash, payload_json
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    order_id, timestamp, display_time,
                    json.dumps(expected), json.dumps(detected),
                    fill_pct, min_fill, json.dumps(missing),
                    status, proof_hash, json.dumps(raw)
                )
            )
            db_id = cur.fetchone()[0]

    # Calculate match confidence score of expected food vs uploaded food
    raw_dets = raw.get("detections", []) if isinstance(raw, dict) else []
    quality_score = calculate_match_confidence(expected, raw_dets)

    # Asynchronously update food_quality_score in 5-minute interval metrics
    try:
        threading.Thread(
            target=update_food_quality_metric,
            kwargs={"dt": now_ist, "current_score": quality_score},
            daemon=True
        ).start()
    except Exception as e:
        print(f"⚠️ Could not start metric update thread: {e}")

    return {"dbId": db_id, "timestamp": timestamp, "time": display_time, "proofHash": proof_hash}


@app.patch("/orders/{db_id}")
def update_order_status(db_id: int, payload: dict = Body(...)):
    """Update order status on Human Verification."""
    new_status = str(payload.get("status", "")).strip()
    if not new_status:
        raise HTTPException(status_code=400, detail="Missing status")

    with closing(get_db()) as conn, conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM dispatch_orders WHERE id = %s", (db_id,))
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Order not found")

            raw = json.loads(row["payload_json"])
            if "verification" in raw:
                raw["verification"]["status"] = new_status

            detected = json.loads(row["detected_json"])
            img_b64 = raw.get("annotated_image", "") or raw.get("original_image", "")

            new_hash = compute_proof_hash(
                row["order_id"], row["timestamp"], row["fill_pct"],
                detected, new_status, img_b64
            )

            cur.execute(
                """
                UPDATE dispatch_orders
                SET status = %s, proof_hash = %s, payload_json = %s
                WHERE id = %s
                """,
                (new_status, new_hash, json.dumps(raw), db_id)
            )

    return {"dbId": db_id, "status": new_status, "proofHash": new_hash}


@app.delete("/orders/{identifier}")
def delete_order(identifier: str):
    """Delete an order from PostgreSQL by either integer id or string order_id."""
    with closing(get_db()) as conn, conn:
        with conn.cursor() as cur:
            if identifier.isdigit():
                cur.execute("DELETE FROM dispatch_orders WHERE id = %s OR order_id = %s", (int(identifier), identifier))
            else:
                cur.execute("DELETE FROM dispatch_orders WHERE order_id = %s", (identifier,))
    return {"deleted": True, "identifier": identifier}


@app.delete("/orders")
def clear_all_orders():
    """Clear all records from dispatch_orders table."""
    with closing(get_db()) as conn, conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM dispatch_orders")
    return {"cleared": True}


@app.post("/fetch-csv")
async def fetch_csv_urls(file: UploadFile = File(...)):
    """Parse an uploaded CSV and persist it in PostgreSQL until manually closed."""
    try:
        raw_bytes = await file.read()
        text = raw_bytes.decode("utf-8-sig", errors="ignore")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read CSV file.")

    reader = csv.reader(io.StringIO(text))
    rows = [r for r in reader if any(cell.strip() for cell in r)]
    if not rows:
        raise HTTPException(status_code=400, detail="CSV file is empty.")

    headers = [h.strip().lower() for h in rows[0]]
    has_header = any(
        h in ("image_url", "url", "link", "img", "image", "item_name", "item_id")
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

        item_id = r[id_idx].strip() if id_idx is not None and id_idx < len(r) else ""
        items.append({"id": item_id, "name": item_name or f"Item {len(items) + 1}", "url": img_url})

    if not items:
        raise HTTPException(status_code=400, detail="No valid image URLs found in CSV.")

    filename = file.filename or "menu.csv"
    with closing(get_db()) as conn, conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO csv_store (id, filename, items_json)
                VALUES (1, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    filename = EXCLUDED.filename,
                    items_json = EXCLUDED.items_json
                """,
                (filename, json.dumps(items))
            )

    return {"filename": filename, "items": items, "total": len(items)}


@app.get("/csv-items")
def get_saved_csv():
    """Return persisted CSV gallery items if a CSV is stored."""
    with closing(get_db()) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT filename, items_json FROM csv_store WHERE id = 1")
            row = cur.fetchone()

    if not row:
        return {"filename": "", "items": [], "total": 0}

    items = json.loads(row["items_json"])
    return {"filename": row["filename"], "items": items, "total": len(items)}


@app.delete("/csv-items")
def clear_saved_csv():
    """Manually close/remove the persisted CSV file from PostgreSQL."""
    with closing(get_db()) as conn, conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM csv_store WHERE id = 1")
    return {"cleared": True}


@app.get("/proxy-image")
def proxy_image(url: str):
    """Fetch an external image URL via backend."""
    clean_url = url.strip()
    if not clean_url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="Invalid URL")

    try:
        req = urllib.request.Request(clean_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            content_type = resp.headers.get("Content-Type", "image/jpeg").split(";")[0].strip()
            data = resp.read()
        if not content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Not an image")
        return Response(content=data, media_type=content_type, headers={"Cache-Control": "public, max-age=3600"})
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=400, detail="Could not fetch image")


@app.post("/predict-url")
async def predict_from_url(url: str = Form(...), conf: float = Form(0.25)):
    """Fetch a reference image from a URL or extract images from a page."""
    clean_url = url.strip()
    if not clean_url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="URL must start with http:// or https://")

    try:
        req = urllib.request.Request(clean_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            content_type = resp.headers.get("Content-Type", "").lower()
            raw_bytes = resp.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not fetch URL: {str(e)}")

    if "text/html" in content_type:
        html = raw_bytes.decode("utf-8", errors="ignore")
        raw_srcs = re.findall(r'<img[^>]+src=["\']([^"\']+)["\']', html, flags=re.IGNORECASE)
        images, seen = [], set()
        for src in raw_srcs:
            abs_url = urljoin(clean_url, src)
            low = abs_url.lower()
            if abs_url in seen or low.endswith(".svg") or "favicon" in low:
                continue
            seen.add(abs_url)
            images.append(abs_url)
        return {"type": "gallery", "images": images[:20]}

    try:
        image = ImageOps.exif_transpose(Image.open(io.BytesIO(raw_bytes))).convert("RGB")
        del raw_bytes
        image.thumbnail((800, 800))
    except Exception:
        raise HTTPException(status_code=400, detail="URL did not return a valid image.")

    detections, _ = run_hybrid_inference(
        image, float(conf), mode="hybrid", expected_list=[], with_plot=False, is_live=False
    )
    fill_pct = float(estimate_fill_percentage(image, detections))

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
        image = ImageOps.exif_transpose(Image.open(io.BytesIO(img_bytes))).convert("RGB")
        del img_bytes
        if live:
            image.thumbnail((256, 256))
        else:
            image.thumbnail((800, 800))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    expected_list = [item.strip() for item in expected.split(",") if item.strip()]
    original_b64 = "" if live else to_base64(image)

    detections, annotated_b64 = run_hybrid_inference(
        image, conf_val, mode=mode, expected_list=expected_list,
        with_plot=(not live), is_live=live
    )

    fill_pct = float(estimate_fill_percentage(image, detections))
    min_fill_val = float(min_fill)
    quantity_ok = bool(fill_pct >= min_fill_val)
    conf_val = float(conf)

    speed_ms = float(round((time.perf_counter() - start) * 1000, 1))
    detected_names = {d["name"] for d in detections}

    matched = [item for item in expected_list if is_expected_matched(item, detected_names)]
    missing = [item for item in expected_list if item not in matched]
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

    # Calculate match confidence score of expected food vs uploaded food
    quality_score = calculate_match_confidence(expected_list, detections)

    # Update 5-minute food_quality_score in metrics DB for every tested food (skip live frames)
    if not live:
        try:
            threading.Thread(
                target=update_food_quality_metric,
                kwargs={"dt": get_current_ist_datetime(), "current_score": quality_score},
                daemon=True
            ).start()
        except Exception as e:
            print(f"⚠️ Could not start metric update thread on predict: {e}")

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
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)