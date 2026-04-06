"""
AR LANGUAGES LENS - QUIZ SYSTEM
Phiên bản cuối cùng - Tối ưu toàn diện
"""

from ultralytics import YOLO
import cv2
import cvzone
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import time
import threading
import pygame
from gtts import gTTS
import os
import tempfile

# ==================== CẤU HÌNH TOÀN CỤC ====================
class Config:
    # Model & Detection
    MODEL_PATH = "1.pt"
    CONFIDENCE_THRESHOLD = 0.7

    # Camera & Display
    CAMERA_WIDTH = 1280
    CAMERA_HEIGHT = 720
    PANEL_WIDTH = 320
    DETECTION_BOX_RATIO = 0.85

    # Timing
    DETECTION_COOLDOWN = 2.0
    OBJECT_LOST_TIMEOUT = 5.0
    FEEDBACK_DISPLAY_TIME = 2.5

    # UI
    FONT_PATH = "arial.ttf"
    USE_VIDEO = False
    VIDEO_PATH = "demo_video.mp4"

# ==================== DATABASE TỪ VỰNG ====================
VOCAB_DB = {
    # Người & Động vật
    "Person": {"vi": "Con người", "answer": "person", "cat": "👤 Người"},
    "Bird": {"vi": "Chim", "answer": "bird", "cat": "🦜 Động vật"},
    "Cat": {"vi": "Mèo", "answer": "cat", "cat": "🦜 Động vật"},
    "Dog": {"vi": "Chó", "answer": "dog", "cat": "🦜 Động vật"},
    "Horse": {"vi": "Ngựa", "answer": "horse", "cat": "🦜 Động vật"},
    "Sheep": {"vi": "Cừu", "answer": "sheep", "cat": "🦜 Động vật"},
    "Cow": {"vi": "Bò", "answer": "cow", "cat": "🦜 Động vật"},
    "Elephant": {"vi": "Voi", "answer": "elephant", "cat": "🦜 Động vật"},
    "Bear": {"vi": "Gấu", "answer": "bear", "cat": "🦜 Động vật"},
    "Zebra": {"vi": "Ngựa vằn", "answer": "zebra", "cat": "🦜 Động vật"},
    "Giraffe": {"vi": "Hươu cao cổ", "answer": "giraffe", "cat": "🦜 Động vật"},

    # Phương tiện
    "Bicycle": {"vi": "Xe đạp", "answer": "bicycle", "cat": "🚗 Phương tiện"},
    "Car": {"vi": "Ô tô", "answer": "car", "cat": "🚗 Phương tiện"},
    "Motorbike": {"vi": "Xe máy", "answer": "motorbike", "cat": "🚗 Phương tiện"},
    "Aeroplane": {"vi": "Máy bay", "answer": "aeroplane", "cat": "🚗 Phương tiện"},
    "Bus": {"vi": "Xe buýt", "answer": "bus", "cat": "🚗 Phương tiện"},
    "Train": {"vi": "Tàu hỏa", "answer": "train", "cat": "🚗 Phương tiện"},
    "Truck": {"vi": "Xe tải", "answer": "truck", "cat": "🚗 Phương tiện"},
    "Boat": {"vi": "Thuyền", "answer": "boat", "cat": "🚗 Phương tiện"},

    # Đồ dùng
    "Cell Phone": {"vi": "Điện thoại", "answer": "cell phone", "cat": "📱 Đồ dùng"},
    "Bottle": {"vi": "Chai", "answer": "bottle", "cat": "📱 Đồ dùng"},
    "Cup": {"vi": "Cốc", "answer": "cup", "cat": "📱 Đồ dùng"},
    "Remote": {"vi": "Điều khiển", "answer": "remote", "cat": "📱 Đồ dùng"}
}

# YOLO Classes
YOLO_CLASSES = [
    "Person", "Bicycle", "Car", "Motorbike", "Aeroplane", "Bus", "Train", "Truck", "Boat",
    "Traffic Light", "Fire Hydrant", "Stop Sign", "Parking Meter", "Bench", "Bird", "Cat",
    "Dog", "Horse", "Sheep", "Cow", "Elephant", "Bear", "Zebra", "Giraffe", "Backpack",
    "Umbrella", "Handbag", "Tie", "Suitcase", "Frisbee", "Skis", "Snowboard", "Sports Ball",
    "Kite", "Baseball Bat", "Baseball Glove", "Skateboard", "Surfboard", "Tennis Racket",
    "Bottle", "Wine Glass", "Cup", "Fork", "Knife", "Spoon", "Bowl", "Banana", "Apple",
    "Sandwich", "Orange", "Broccoli", "Carrot", "Hot Dog", "Pizza", "Donut", "Cake",
    "Chair", "Sofa", "Potted Plant", "Bed", "Dining Table", "Toilet", "TV Monitor",
    "Laptop", "Mouse", "Remote", "Keyboard", "Cell Phone", "Microwave", "Oven",
    "Toaster", "Sink", "Refrigerator", "Book", "Clock", "Vase", "Scissors",
    "Teddy Bear", "Hair Drier", "Toothbrush"
]

# ==================== QUIZ STATE ====================
class QuizState:
    def __init__(self):
        self.current_object = None
        self.current_question = None
        self.user_answer = ""
        self.feedback = ""
        self.feedback_time = 0
        self.score = 0
        self.total = 0
        self.waiting = False
        self.show_feedback = False
        self.q_start_time = 0
        self.answer_time = 0
        self.last_seen_time = 0
        self.show_hint = False
        self.last_spoken = None
        self.speaking = False
        self.question_count = {}  # Đếm số lần hỏi mỗi vật

    def reset_question(self):
        self.current_question = None
        self.user_answer = ""
        self.feedback = ""
        self.waiting = False
        self.show_feedback = False
        self.q_start_time = 0
        self.answer_time = 0
        self.show_hint = False

    def start_question(self, obj_name):
        if obj_name not in VOCAB_DB:
            return False

        # Đếm số lần hỏi vật này
        if obj_name not in self.question_count:
            self.question_count[obj_name] = 0
        self.question_count[obj_name] += 1

        # CHỈ phát âm nếu:
        # - Lần đầu tiên (count = 1)
        # - Hoặc cứ mỗi 3 lần (count % 3 == 0) để không lặp quá nhiều
        should_speak = (self.question_count[obj_name] == 1 or
                       self.question_count[obj_name] % 3 == 0)

        self.current_object = obj_name
        self.current_question = VOCAB_DB[obj_name]["vi"]
        self.user_answer = ""
        self.feedback = ""
        self.waiting = True
        self.show_feedback = False
        self.q_start_time = time.time()
        self.answer_time = 0
        self.show_hint = False

        # Phát âm CÂU HỎI khi panel màu cam hiện (nếu cần)
        if should_speak:
            vi_name = VOCAB_DB[obj_name]["vi"]
            speak_vietnamese(f"{vi_name} trong tiếng Anh là gì")

        return True

    def get_hint(self):
        if not self.current_object or self.current_object not in VOCAB_DB:
            return ""
        answer = VOCAB_DB[self.current_object]["answer"]
        return " ".join([w[0] + "_" * (len(w) - 1) for w in answer.split()])

    def submit(self):
        if not self.user_answer or not self.current_object:
            return False
        if self.current_object not in VOCAB_DB:
            return False

        self.answer_time = int(time.time() - self.q_start_time)
        correct_answer = VOCAB_DB[self.current_object]["answer"]
        vi_name = VOCAB_DB[self.current_object]["vi"]

        self.total += 1

        if self.user_answer.lower().strip() == correct_answer.lower().strip():
            self.score += 1
            self.feedback = "✅ CORRECT!"
            # Phát âm khi ĐÚNG: Tiếng Việt + Tiếng Anh chuẩn
            speak_vietnamese_and_english(f"Chính xác, {vi_name} là", correct_answer)
        else:
            self.feedback = f"❌ WRONG! → {correct_answer}"
            # Phát âm khi SAI: Tiếng Việt + Tiếng Anh chuẩn
            speak_vietnamese_and_english(f"Chưa chính xác, {vi_name} là", correct_answer)

        self.show_feedback = True
        self.feedback_time = time.time()
        self.waiting = False
        return True

quiz = QuizState()

# ==================== GIỌNG NÓI ====================
# Khởi tạo pygame mixer
pygame.mixer.init()

def speak_vietnamese_and_english(vi_text, en_text=None):
    """
    Phát âm tiếng Việt + tiếng Anh (nếu có)
    Đảm bảo KHÔNG BỊ HỤT GIỌNG
    """
    def _speak():
        try:
            # Chờ nếu đang nói
            while quiz.speaking:
                time.sleep(0.1)

            quiz.speaking = True

            # Phần 1: Tiếng Việt
            temp_vi = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
            temp_vi_path = temp_vi.name
            temp_vi.close()

            tts_vi = gTTS(text=vi_text, lang='vi', slow=False)
            tts_vi.save(temp_vi_path)

            # Đợi file được tạo xong
            time.sleep(0.1)

            pygame.mixer.music.load(temp_vi_path)
            pygame.mixer.music.play()

            # Chờ phát xong
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)

            # Dọn dẹp
            pygame.mixer.music.unload()
            time.sleep(0.1)
            try:
                os.remove(temp_vi_path)
            except:
                pass

            # Phần 2: Tiếng Anh (nếu có) - PHÁT ÂM CỰC CHUẨN
            if en_text:
                time.sleep(0.3)  # Nghỉ giữa 2 ngôn ngữ

                temp_en = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
                temp_en_path = temp_en.name
                temp_en.close()

                # Dùng giọng Anh UK hoặc US (chuẩn nhất)
                tts_en = gTTS(text=en_text, lang='en', slow=False, tld='com')
                tts_en.save(temp_en_path)

                # Đợi file được tạo xong
                time.sleep(0.1)

                pygame.mixer.music.load(temp_en_path)
                pygame.mixer.music.play()

                # Chờ phát xong
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)

                # Dọn dẹp
                pygame.mixer.music.unload()
                time.sleep(0.1)
                try:
                    os.remove(temp_en_path)
                except:
                    pass

            quiz.speaking = False

        except Exception as e:
            print(f"⚠️ Voice error: {e}")
            quiz.speaking = False

    threading.Thread(target=_speak, daemon=True).start()

def speak_vietnamese(text):
    """Chỉ phát âm tiếng Việt"""
    speak_vietnamese_and_english(text, None)

# ==================== UI DRAWING ====================
def draw_vn_text(img, text, pos, size=24, color=(255, 255, 255)):
    try:
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        try:
            font = ImageFont.truetype(Config.FONT_PATH, size)
        except:
            font = ImageFont.load_default()
        draw.text(pos, text, font=font, fill=color)
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except:
        return img

def draw_detection_box(img, coords):
    x1, y1, x2, y2 = coords
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 4)

    # Góc nhấn mạnh
    L = 40
    T = 6
    C = (0, 255, 255)
    # Top-left
    cv2.line(img, (x1, y1+L), (x1, y1), C, T)
    cv2.line(img, (x1, y1), (x1+L, y1), C, T)
    # Top-right
    cv2.line(img, (x2-L, y1), (x2, y1), C, T)
    cv2.line(img, (x2, y1), (x2, y1+L), C, T)
    # Bottom-left
    cv2.line(img, (x1, y2-L), (x1, y2), C, T)
    cv2.line(img, (x1, y2), (x1+L, y2), C, T)
    # Bottom-right
    cv2.line(img, (x2-L, y2), (x2, y2), C, T)
    cv2.line(img, (x2, y2-L), (x2, y2), C, T)

    img = draw_vn_text(img, "Đặt vật ở đây", (x1 + 20, y1 - 35), 22, (0, 255, 255))
    return img

def draw_panel(h, w=500):
    panel = np.zeros((h, w, 3), dtype=np.uint8)
    panel[:] = (45, 45, 45)
    cv2.rectangle(panel, (0, 0), (w-1, h-1), (0, 200, 255), 3)

    y = 30
    panel = draw_vn_text(panel, "🎮 AR QUIZ", (20, y), 32, (0, 255, 255))
    y += 50
    cv2.line(panel, (20, y), (w-20, y), (100, 100, 100), 2)
    y += 25

    score_txt = f"SCORE: {quiz.score}/{quiz.total}"
    panel = draw_vn_text(panel, score_txt, (20, y), 26, (0, 255, 0))
    y += 45

    if quiz.current_question:
        # Question box
        qh = 90
        cv2.rectangle(panel, (15, y), (w-15, y+qh), (60, 120, 180), -1)
        cv2.rectangle(panel, (15, y), (w-15, y+qh), (100, 150, 255), 2)
        panel = draw_vn_text(panel, f"{quiz.current_question} ?", (25, y+15), 22, (255, 255, 255))
        panel = draw_vn_text(panel, "in English?", (25, y+50), 18, (200, 200, 200))
        y += qh + 25

        # Timer
        if quiz.waiting:
            elapsed = int(time.time() - quiz.q_start_time)
            tc = (0, 255, 0) if elapsed < 10 else (255, 150, 0) if elapsed < 20 else (0, 0, 255)
            panel = draw_vn_text(panel, f"⏱️ Time: {elapsed}s", (20, y), 22, tc)
            y += 35

        # Hint
        if quiz.show_hint:
            panel = draw_vn_text(panel, f"💡 Hint: {quiz.get_hint()}", (20, y), 20, (255, 255, 100))
            y += 35

        # Input box
        panel = draw_vn_text(panel, "YOUR ANSWER:", (20, y), 20, (255, 255, 0))
        y += 30
        ih = 55
        cv2.rectangle(panel, (15, y), (w-15, y+ih), (30, 30, 30), -1)
        cv2.rectangle(panel, (15, y), (w-15, y+ih), (0, 255, 0), 3)
        txt = quiz.user_answer
        if quiz.waiting and int(time.time() * 2) % 2:
            txt += "|"
        panel = draw_vn_text(panel, txt, (25, y+15), 24, (255, 255, 255))
        y += ih + 25

        # Feedback
        if quiz.show_feedback and quiz.feedback:
            fh = 100
            bg = (0, 100, 0) if "✅" in quiz.feedback else (100, 0, 0)
            bd = (0, 255, 0) if "✅" in quiz.feedback else (0, 0, 255)
            cv2.rectangle(panel, (15, y), (w-15, y+fh), bg, -1)
            cv2.rectangle(panel, (15, y), (w-15, y+fh), bd, 3)
            panel = draw_vn_text(panel, quiz.feedback, (25, y+18), 22, (255, 255, 255))
            if quiz.answer_time > 0:
                tc = (150, 255, 150) if quiz.answer_time < 10 else (255, 255, 150)
                panel = draw_vn_text(panel, f"⏱️ {quiz.answer_time}s", (25, y+55), 18, tc)
            y += fh + 20

        # Controls
        y = h - 180
        panel = draw_vn_text(panel, "CONTROLS:", (20, y), 18, (200, 200, 200))
        y += 28
        panel = draw_vn_text(panel, "ENTER - Submit", (20, y), 16, (150, 150, 150))
        y += 23
        panel = draw_vn_text(panel, "TAB - Hint (Gợi ý)", (20, y), 16, (255, 255, 100))
        y += 23
        panel = draw_vn_text(panel, "BACKSPACE - Delete", (20, y), 16, (150, 150, 150))
        y += 23
        panel = draw_vn_text(panel, "Q - Quit", (20, y), 16, (150, 150, 150))
    else:
        panel = draw_vn_text(panel, "Hướng camera vào", (20, y), 22, (200, 200, 200))
        y += 30
        panel = draw_vn_text(panel, "vật thể...", (20, y), 22, (200, 200, 200))
        y += 45

        # Categories
        cats = {}
        for obj, info in VOCAB_DB.items():
            c = info.get("cat", "Khác")
            if c not in cats:
                cats[c] = []
            cats[c].append((obj, info["vi"]))

        for cat in ["👤 Người", "🦜 Động vật", "🚗 Phương tiện", "📱 Đồ dùng"]:
            if cat not in cats:
                continue
            panel = draw_vn_text(panel, cat, (20, y), 18, (255, 255, 0))
            y += 28
            for obj, vi in sorted(cats[cat], key=lambda x: x[1]):
                emoji = "✓" if obj == quiz.current_object else "○"
                col = (0, 255, 0) if obj == quiz.current_object else (150, 150, 150)
                name = vi if len(vi) <= 15 else vi[:13] + ".."
                panel = draw_vn_text(panel, f"{emoji} {name}", (30, y), 16, col)
                y += 23
                if y > h - 50:
                    break
            y += 5
            if y > h - 50:
                break

    return panel

# ==================== KEYBOARD ====================
def handle_key(key):
    if quiz.waiting:
        if key == 13:  # ENTER
            quiz.submit()
        elif key == 9:  # TAB - Gợi ý
            quiz.show_hint = not quiz.show_hint
        elif key == 8:  # BACKSPACE
            quiz.user_answer = quiz.user_answer[:-1]
        elif 32 <= key <= 126:  # Ký tự thông thường
            quiz.user_answer += chr(key)

# ==================== MAIN ====================
def main():
    # Init camera
    if Config.USE_VIDEO:
        cap = cv2.VideoCapture(Config.VIDEO_PATH)
        print(f"✅ Video: {Config.VIDEO_PATH}")
    else:
        cap = None
        for i in [0, 1, 2]:
            temp = cv2.VideoCapture(i)
            if temp.isOpened():
                cap = temp
                print(f"✅ Camera: index {i}")
                break
            temp.release()

        if not cap:
            print("❌ No camera found!")
            return

    cap.set(3, Config.CAMERA_WIDTH)
    cap.set(4, Config.CAMERA_HEIGHT)

    # Load model
    try:
        model = YOLO(Config.MODEL_PATH)
        model.fuse()
        print("✅ Model loaded")
    except Exception as e:
        print(f"❌ Model error: {e}")
        return

    print("\n" + "="*70)
    print("🎓 AR LANGUAGES LENS - FINAL VERSION")
    print("="*70)
    print("📷 23 vật: Người, Động vật, Phương tiện, Đồ dùng")
    print(f"🎯 Confidence: {Config.CONFIDENCE_THRESHOLD}")
    print(f"📺 Resolution: {Config.CAMERA_WIDTH}x{Config.CAMERA_HEIGHT}")
    print("🔊 Giọng nói tự động")
    print("💡 Nhấn Tab để xem gợi ý")
    print("="*70 + "\n")

    last_detect = 0

    while True:
        ret, img = cap.read()
        if not ret:
            break

        try:
            # Resize
            cam_w = Config.CAMERA_WIDTH - Config.PANEL_WIDTH
            img = cv2.resize(img, (cam_w, Config.CAMERA_HEIGHT))
            h, w, _ = img.shape

            # Detection box
            m = int((1 - Config.DETECTION_BOX_RATIO) / 2 * min(w, h))
            box = (m, m, w - m, h - m)
            img = draw_detection_box(img, box)

            # Detect
            now = time.time()
            detected = False
            best = None
            max_conf = 0

            results = model(img, stream=True, verbose=False, imgsz=640)

            for r in results:
                for box_data in r.boxes:
                    try:
                        x1, y1, x2, y2 = map(int, box_data.xyxy[0])
                        conf = float(box_data.conf[0])

                        if conf < Config.CONFIDENCE_THRESHOLD:
                            continue

                        cls = int(box_data.cls[0])
                        if cls >= len(YOLO_CLASSES):
                            continue

                        obj = YOLO_CLASSES[cls]
                        if obj not in VOCAB_DB:
                            continue

                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                        in_box = (box[0] < cx < box[2] and box[1] < cy < box[3])

                        if in_box and conf > max_conf:
                            max_conf = conf
                            best = {'obj': obj, 'conf': conf, 'box': (x1, y1, x2, y2)}
                    except:
                        continue

            # Process best detection
            if best:
                detected = True
                quiz.last_seen_time = now

                obj = best['obj']
                x1, y1, x2, y2 = best['box']

                # Draw box
                w_box, h_box = x2 - x1, y2 - y1
                cvzone.cornerRect(img, (x1, y1, w_box, h_box), colorR=(0, 255, 0), colorC=(0, 255, 0))

                # Draw label - GÓC TRÊN TRÁI
                vi_name = VOCAB_DB[obj]["vi"]
                tx1, ty1 = x1 + 5, y1 + 5
                tx2, ty2 = tx1 + len(vi_name) * 20 + 20, ty1 + 45

                overlay = img.copy()
                cv2.rectangle(overlay, (tx1, ty1), (tx2, ty2), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
                cv2.rectangle(img, (tx1, ty1), (tx2, ty2), (0, 255, 255), 2)
                img = draw_vn_text(img, vi_name, (tx1 + 10, ty1 + 8), 26, (0, 255, 255))

                # Update state
                if not quiz.waiting:
                    # CHỈ cập nhật object, KHÔNG phát âm ở đây
                    if obj != quiz.current_object:
                        quiz.current_object = obj
                        quiz.last_spoken = obj

                # New question - Giọng nói sẽ phát trong start_question()
                if (not quiz.waiting and not quiz.show_feedback and
                    now - last_detect > Config.DETECTION_COOLDOWN):
                    if quiz.start_question(obj):
                        last_detect = now

            # Handle lost object
            if not detected and quiz.waiting:
                if now - quiz.last_seen_time > Config.OBJECT_LOST_TIMEOUT:
                    quiz.reset_question()
                    quiz.current_object = None

            # Auto hide feedback
            if quiz.show_feedback and now - quiz.feedback_time > Config.FEEDBACK_DISPLAY_TIME:
                quiz.reset_question()

            # Draw panel
            panel = draw_panel(Config.CAMERA_HEIGHT, Config.PANEL_WIDTH)

            # Combine
            full = np.hstack([img, panel])
            cv2.imshow("AR Languages Lens - Quiz", full)

            # Keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            else:
                handle_key(key)

        except Exception as e:
            print(f"⚠️ Error: {e}")
            continue

    cap.release()
    cv2.destroyAllWindows()

    # Report
    print("\n" + "="*70)
    print("📈 FINAL RESULTS")
    print("="*70)
    print(f"🎯 Score: {quiz.score}/{quiz.total}")
    if quiz.total > 0:
        acc = (quiz.score / quiz.total) * 100
        print(f"✨ Accuracy: {acc:.1f}%")
    print("="*70)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Stopped by user")
    except Exception as e:
        print(f"\n❌ Critical error: {e}")
        import traceback
        traceback.print_exc()