"""
剪绳子互动游戏 / Rope Cut Interactive Game
============================================
用途: 录视频时做出剪刀手势 (✌️), 摄像头识别后触发屏幕里绳子断开、人物坠落的效果。

依赖:
    pip install pygame opencv-python mediapipe numpy

运行:
    python rope_cut_game.py

操作:
    - 对摄像头做剪刀手势 (食指+中指伸直) 并对准绳子位置
    - 空格键: 手动触发剪断 (测试用)
    - R 键:   重置场景
    - ESC:    退出
"""

import os
import sys
import math
import random
import time
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np
import pygame

# ---------------------------------------------------------------------------
# 首次运行自动下载 MediaPipe 手部检测模型 (~10 MB)
# ---------------------------------------------------------------------------
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
_MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)

def _ensure_model():
    if not os.path.exists(_MODEL_PATH):
        print("首次运行：正在下载手部检测模型 (~10 MB)...")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        print("模型下载完成！")

_ensure_model()


# ---------------------------------------------------------------------------
# 游戏状态常量
# ---------------------------------------------------------------------------
STATE_HANGING = "hanging"   # 人物悬挂中
STATE_CUTTING = "cutting"   # 剪断瞬间特效
STATE_FALLING = "falling"   # 人物坠落中
STATE_RESET    = "reset"    # 重置倒计时


# ---------------------------------------------------------------------------
# 颜色 — 暗黑地牢主题
# ---------------------------------------------------------------------------
BG_TOP      = (8,    6,   14)   # 顶部近黑
BG_MID      = (18,  12,   28)   # 中部深紫黑
BG_BOT      = (10,   8,   18)   # 底部近黑
STONE_DARK  = (30,  28,   32)
STONE_MID   = (48,  44,   52)
STONE_LIGHT = (68,  62,   72)
IRON_COL    = (55,  55,   60)
ROPE_DARK   = (60,  38,   18)
ROPE_MID    = (90,  58,   25)
ROPE_LIGHT  = (130, 88,   40)
TORCH_WARM  = (255, 140,  30)
TORCH_HOT   = (255, 220, 100)
BLOOD_RED   = (180,  20,  20)
SPARK_COL   = (255, 180,  40)
RIM_LIGHT   = (200, 130,  50)   # 火把照出的轮廓光
WHITE       = (255, 255, 255)
RED         = (200,  30,  30)
GREEN       = (40,  200,  80)
GREY_DIM    = (90,  85,  100)


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def lerp(a, b, t):
    return a + (b - a) * t


def draw_gradient_rect(surface, color_top, color_bot, rect):
    x, y, w, h = rect
    for i in range(h):
        t = i / max(h - 1, 1)
        c = (
            int(lerp(color_top[0], color_bot[0], t)),
            int(lerp(color_top[1], color_bot[1], t)),
            int(lerp(color_top[2], color_bot[2], t)),
        )
        pygame.draw.line(surface, c, (x, y + i), (x + w, y + i))


# ---------------------------------------------------------------------------
# 游戏主类
# ---------------------------------------------------------------------------
class RopeCutGame:

    # 布局尺寸
    SCREEN_W = 1200
    SCREEN_H = 700
    GAME_W   = 760   # 左侧游戏画面宽度
    CAM_W    = 440   # 右侧摄像头宽度

    # 绳子参数
    ROPE_ATTACH_X = 380   # 绳子顶端 x (游戏坐标)
    ROPE_ATTACH_Y =  60   # 绳子顶端 y
    ROPE_LENGTH   = 280   # 绳子长度 (像素)

    # 物理
    GRAVITY       = 0.55
    SWING_SPEED   = 0.018
    SWING_AMPLITUDE = 32  # 最大摆动幅度 (像素)

    # 剪断特效帧数
    CUT_FLASH_FRAMES = 22

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((self.SCREEN_W, self.SCREEN_H))
        pygame.display.set_caption("✂  剪绳子互动游戏  ✂")
        self.clock = pygame.time.Clock()

        # 字体
        self._init_fonts()

        # 摄像头 + MediaPipe Tasks API
        self.cap = cv2.VideoCapture(0)
        base_opts = mp_python.BaseOptions(model_asset_path=_MODEL_PATH)
        opts = mp_vision.HandLandmarkerOptions(
            base_options=base_opts,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.75,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        self.hand_landmarker = mp_vision.HandLandmarker.create_from_options(opts)
        self._frame_ts = 0  # 递增时间戳 (毫秒)

        # 游戏状态
        self._reset_state()

        # 预渲染静态背景
        self._bg_surface = None

    # ------------------------------------------------------------------
    # 初始化辅助
    # ------------------------------------------------------------------
    def _init_fonts(self):
        # 尝试加载支持中文的系统字体, 否则回退到默认
        candidates = [
            "notosanscjk", "noto sans cjk sc", "wqy microhei",
            "simhei", "microsoft yahei", "arialunicodems",
        ]
        font_name = None
        for name in candidates:
            if name in [f.lower() for f in pygame.font.get_fonts()]:
                font_name = name
                break

        self.font_lg  = pygame.font.SysFont(font_name, 68, bold=True)
        self.font_md  = pygame.font.SysFont(font_name, 46, bold=True)
        self.font_sm  = pygame.font.SysFont(font_name, 30)
        self.font_tip = pygame.font.SysFont(font_name, 24)

    def _reset_state(self):
        self.state           = STATE_HANGING
        self.swing_angle     = 0.0
        self.char_x          = float(self.ROPE_ATTACH_X)
        self.char_y          = float(self.ROPE_ATTACH_Y + self.ROPE_LENGTH)
        self.char_vx         = 0.0
        self.char_vy         = 0.0
        self.cut_y           = 0.0
        self.cut_flash       = 0
        self.reset_timer     = 0
        self.scissors_pos    = None
        self.scissors_active = False
        self.frays           = []
        # 暗黑风特效
        self.sparks          = []           # 火花粒子 [(x,y,vx,vy,life,max_life)]
        self.torch_phase     = 0.0          # 火把闪烁相位
        self.speed_lines     = []           # 坠落速度线

    # ------------------------------------------------------------------
    # 摄像头 & 手势处理
    # ------------------------------------------------------------------
    def _process_camera(self):
        ret, frame = self.cap.read()
        if not ret:
            return None

        frame = cv2.flip(frame, 1)          # 镜像, 让用户看起来自然
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Tasks API 需要递增时间戳
        self._frame_ts += 33
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = self.hand_landmarker.detect_for_video(mp_image, self._frame_ts)

        self.scissors_active = False
        self.scissors_pos    = None

        if res.hand_landmarks:
            lm = res.hand_landmarks[0]      # list of NormalizedLandmark
            if self._is_scissors(lm):
                self.scissors_active = True
                self.scissors_pos    = self._hand_to_game_pos(lm)

        return frame

    @staticmethod
    def _is_scissors(lm):
        """检测剪刀手势: 食指+中指伸直, 无名指+小指弯曲"""
        index_up  = lm[8].y  < lm[6].y   # 食指伸直
        middle_up = lm[12].y < lm[10].y  # 中指伸直
        ring_dn   = lm[16].y > lm[14].y  # 无名指弯曲
        pinky_dn  = lm[20].y > lm[18].y  # 小指弯曲
        return index_up and middle_up and ring_dn and pinky_dn

    def _hand_to_game_pos(self, lm):
        """将手部归一化坐标映射到游戏画面坐标"""
        # 取食指和中指指尖的中点作为剪刀位置
        hx = (lm[8].x + lm[12].x) / 2
        hy = (lm[8].y + lm[12].y) / 2
        # 摄像头已翻转, 直接线性映射
        gx = hx * self.GAME_W
        gy = hy * self.SCREEN_H
        return (gx, gy)

    # ------------------------------------------------------------------
    # 游戏逻辑更新
    # ------------------------------------------------------------------
    def _rope_x_at_y(self, y):
        """绳上某 y 处的 x (含摆动偏移)"""
        t = (y - self.ROPE_ATTACH_Y) / max(self.ROPE_LENGTH, 1)
        offset = math.sin(self.swing_angle) * self.SWING_AMPLITUDE * t
        return self.ROPE_ATTACH_X + offset

    def _scissors_near_rope(self):
        """判断剪刀位置是否贴近绳子"""
        if not self.scissors_pos:
            return False
        sx, sy = self.scissors_pos
        ry_min = self.ROPE_ATTACH_Y + 10
        ry_max = self.ROPE_ATTACH_Y + self.ROPE_LENGTH - 10
        if not (ry_min < sy < ry_max):
            return False
        rx = self._rope_x_at_y(sy)
        return abs(sx - rx) < 55

    def _trigger_cut(self, cut_y=None):
        if cut_y is None:
            cut_y = self.ROPE_ATTACH_Y + self.ROPE_LENGTH * 0.5
        self.state     = STATE_CUTTING
        self.cut_y     = cut_y
        self.cut_flash = self.CUT_FLASH_FRAMES
        # 记录角色当前位置和速度 (继承摆动速度)
        swing_offset = math.sin(self.swing_angle) * self.SWING_AMPLITUDE
        swing_vx     = (math.cos(self.swing_angle)
                        * self.SWING_AMPLITUDE * self.SWING_SPEED)
        self.char_x  = self.ROPE_ATTACH_X + swing_offset
        self.char_y  = float(self.ROPE_ATTACH_Y + self.ROPE_LENGTH)
        self.char_vx = swing_vx * 2.5
        self.char_vy = 0.0
        # 断口毛刺
        self.frays = [
            (random.randint(-14, 14), random.randint(2, 18))
            for _ in range(8)
        ]
        # 生成火花粒子
        cx_spark = self._rope_x_at_y(cut_y)
        for _ in range(28):
            angle  = random.uniform(0, math.pi * 2)
            speed  = random.uniform(1.5, 6.0)
            life   = random.randint(18, 40)
            self.sparks.append([
                cx_spark, cut_y,
                math.cos(angle) * speed,
                math.sin(angle) * speed - random.uniform(0, 2),
                life, life,
            ])

    def update(self):
        if self.state == STATE_HANGING:
            self.swing_angle += self.SWING_SPEED
            if self.scissors_active and self._scissors_near_rope():
                self._trigger_cut(self.scissors_pos[1])

        elif self.state == STATE_CUTTING:
            self.cut_flash -= 1
            if self.cut_flash <= 0:
                self.state = STATE_FALLING

        elif self.state == STATE_FALLING:
            self.char_vy += self.GRAVITY
            self.char_x  += self.char_vx
            self.char_y  += self.char_vy
            # 轻微空气阻力
            self.char_vx *= 0.995
            if self.char_y > self.SCREEN_H + 150:
                self.state       = STATE_RESET
                self.reset_timer = 150   # ~2.5 秒后重置

        elif self.state == STATE_RESET:
            self.reset_timer -= 1
            if self.reset_timer <= 0:
                self._reset_state()

        # 火把闪烁 & 火花粒子 (每帧更新)
        self.torch_phase += 0.08
        for s in self.sparks:
            s[0] += s[2]; s[1] += s[3]
            s[3] += 0.3        # 重力
            s[2] *= 0.92       # 摩擦
            s[4] -= 1
        self.sparks = [s for s in self.sparks if s[4] > 0]

    # ------------------------------------------------------------------
    # 渲染
    # ------------------------------------------------------------------
    def _build_bg(self):
        """预渲染静态暗黑地牢背景"""
        surf = pygame.Surface((self.GAME_W, self.SCREEN_H))
        # 深暗渐变背景
        draw_gradient_rect(surf, BG_TOP, BG_MID, (0, 0, self.GAME_W, self.SCREEN_H // 2))
        draw_gradient_rect(surf, BG_MID, BG_BOT,
                           (0, self.SCREEN_H // 2, self.GAME_W, self.SCREEN_H // 2))

        # 石砖墙纹理 (横排砖块)
        bw, bh = 80, 36
        for row in range(self.SCREEN_H // bh + 1):
            off = (row % 2) * (bw // 2)
            for col in range(-1, self.GAME_W // bw + 2):
                rx = col * bw + off
                ry = row * bh
                shade = random.randint(-6, 6)
                c = tuple(max(0, min(255, v + shade)) for v in STONE_DARK)
                pygame.draw.rect(surf, c, (rx + 2, ry + 2, bw - 4, bh - 4))
                pygame.draw.rect(surf, STONE_MID, (rx + 2, ry + 2, bw - 4, bh - 4), 1)

        # 天花板厚石板
        pygame.draw.rect(surf, STONE_DARK, (0, 0, self.GAME_W, 58))
        pygame.draw.rect(surf, STONE_MID,  (0, 55, self.GAME_W, 4))
        # 天花板裂缝
        for cx in [120, 290, 500, 650]:
            pygame.draw.line(surf, (15, 12, 18),
                             (cx, 0), (cx + random.randint(-20, 20), 58), 2)

        # 铁钩 (挂绳子的地方)
        hx = self.ROPE_ATTACH_X
        pygame.draw.rect(surf, IRON_COL, (hx - 10, 0, 20, 72))
        pygame.draw.rect(surf, STONE_LIGHT, (hx - 10, 0, 20, 72), 1)
        pygame.draw.circle(surf, IRON_COL, (hx, 72), 10)
        pygame.draw.circle(surf, STONE_LIGHT, (hx, 72), 10, 2)

        # 地面石板
        pygame.draw.rect(surf, STONE_DARK, (0, self.SCREEN_H - 55, self.GAME_W, 55))
        pygame.draw.rect(surf, STONE_MID,  (0, self.SCREEN_H - 55, self.GAME_W, 3))
        for gx in range(0, self.GAME_W, 120):
            pygame.draw.line(surf, STONE_LIGHT,
                             (gx, self.SCREEN_H - 55), (gx, self.SCREEN_H), 1)

        self._bg_surface = surf

    def _draw_rope_segment(self, y_start, y_end, n=20):
        """绘制一段三股麻绳 (暗色+亮色双线营造质感)"""
        pts = []
        for i in range(n + 1):
            t = i / n
            y = lerp(y_start, y_end, t)
            x = self._rope_x_at_y(y)
            pts.append((int(x), int(y)))
        if len(pts) < 2:
            return
        # 阴影层
        pygame.draw.lines(self.screen, ROPE_DARK, False, pts, 9)
        # 主绳
        pygame.draw.lines(self.screen, ROPE_MID, False, pts, 6)
        # 高光 (模拟火把侧光)
        hi = [(p[0] - 1, p[1]) for p in pts]
        pygame.draw.lines(self.screen, ROPE_LIGHT, False, hi, 2)

    def _draw_rope(self):
        if self.state == STATE_HANGING:
            self._draw_rope_segment(self.ROPE_ATTACH_Y, self.ROPE_ATTACH_Y + self.ROPE_LENGTH)

        elif self.state in (STATE_CUTTING, STATE_FALLING, STATE_RESET):
            t_cut = (self.cut_y - self.ROPE_ATTACH_Y) / max(self.ROPE_LENGTH, 1)
            t_cut = max(0.05, min(0.95, t_cut))
            y_cut = self.ROPE_ATTACH_Y + t_cut * self.ROPE_LENGTH
            self._draw_rope_segment(self.ROPE_ATTACH_Y, y_cut)

            # 断口毛刺 (焦黑纤维)
            cx = int(self._rope_x_at_y(self.cut_y))
            cy = int(self.cut_y)
            for fx, fy in self.frays:
                color = ROPE_DARK if abs(fx) > 7 else ROPE_MID
                pygame.draw.line(self.screen, color,
                                 (cx, cy), (cx + fx, cy + fy), 2)

        # 火花粒子
        for s in self.sparks:
            life_ratio = s[4] / s[5]
            r = int(lerp(180, 255, life_ratio))
            g = int(lerp(20,  180, life_ratio))
            b = 0
            radius = max(1, int(life_ratio * 4))
            pygame.draw.circle(self.screen, (r, g, b), (int(s[0]), int(s[1])), radius)

    def _draw_character(self, x, y, falling=False):
        """暗黑风人物: 黑色披风剪影 + 火把轮廓光"""
        ix, iy = int(x), int(y)
        angle  = min(self.char_vy * 4.0, 95) if falling else 0.0

        size = 200
        tmp  = pygame.Surface((size, size), pygame.SRCALPHA)
        cx   = size // 2
        # 头部中心 y (挂绳时双手举过头顶)
        head_y = 52

        # ------ 双手绑在绳子上 ------
        if not falling:
            # 左右手腕绑绳处
            pygame.draw.line(tmp, (80, 55, 25), (cx - 10, head_y - 28),
                             (cx + 10, head_y - 28), 5)  # 绳结
            pygame.draw.line(tmp, (60, 40, 15), (cx, head_y - 35), (cx, head_y - 28), 4)

        # ------ 披风/斗篷 (大三角形剪影) ------
        cloak_top  = head_y + 18
        cloak_bot  = head_y + 140
        cloak_pts  = [
            (cx,      cloak_top),
            (cx - 42, cloak_bot - 20),
            (cx - 30, cloak_bot),
            (cx,      cloak_bot - 15),
            (cx + 30, cloak_bot),
            (cx + 42, cloak_bot - 20),
        ]
        pygame.draw.polygon(tmp, (18, 14, 22), cloak_pts)        # 黑色主体
        # 披风边缘轮廓光
        pygame.draw.lines(tmp, RIM_LIGHT, False,
                          [(cx - 42, cloak_bot - 20),
                           (cx,      cloak_top),
                           (cx + 42, cloak_bot - 20)], 2)

        # ------ 腿 ------
        leg_a = math.sin(self.swing_angle * 2) * 8 if not falling else 28
        pygame.draw.line(tmp, (25, 20, 30),
                         (cx - 8, cloak_bot - 20),
                         (cx - 18, cloak_bot + 35 + int(leg_a)), 7)
        pygame.draw.line(tmp, (25, 20, 30),
                         (cx + 8, cloak_bot - 20),
                         (cx + 18, cloak_bot + 35 - int(leg_a)), 7)
        # 腿部轮廓光
        pygame.draw.line(tmp, RIM_LIGHT,
                         (cx - 8, cloak_bot - 20),
                         (cx - 18, cloak_bot + 35 + int(leg_a)), 1)

        # ------ 头部 ------
        # 头骨阴影 (球体)
        pygame.draw.circle(tmp, (22, 18, 26), (cx, head_y), 26)
        # 轮廓光
        pygame.draw.circle(tmp, RIM_LIGHT, (cx, head_y), 26, 2)
        # 兜帽遮住大半张脸
        hood_pts = [
            (cx - 26, head_y - 4),
            (cx - 20, head_y - 26),
            (cx,      head_y - 32),
            (cx + 20, head_y - 26),
            (cx + 26, head_y - 4),
            (cx + 20, head_y + 10),
            (cx,      head_y + 14),
            (cx - 20, head_y + 10),
        ]
        pygame.draw.polygon(tmp, (18, 14, 22), hood_pts)

        # 眼睛 — 坠落时睁大, 悬挂时暗红光点
        if falling:
            pygame.draw.circle(tmp, (220, 40, 40), (cx - 9, head_y), 6)
            pygame.draw.circle(tmp, (220, 40, 40), (cx + 9, head_y), 6)
            pygame.draw.circle(tmp, (255, 120, 80), (cx - 9, head_y), 3)
            pygame.draw.circle(tmp, (255, 120, 80), (cx + 9, head_y), 3)
        else:
            pygame.draw.circle(tmp, (140, 20, 20), (cx - 8, head_y + 2), 4)
            pygame.draw.circle(tmp, (140, 20, 20), (cx + 8, head_y + 2), 4)

        # 旋转并贴图
        rotated = pygame.transform.rotate(tmp, -angle)
        rw, rh  = rotated.get_size()
        self.screen.blit(rotated, (ix - rw // 2, iy - head_y - rh // 4 + 10))

    def _draw_scissors_cursor(self):
        if not self.scissors_active or not self.scissors_pos:
            return
        sx, sy   = int(self.scissors_pos[0]), int(self.scissors_pos[1])
        on_rope  = self._scissors_near_rope()
        color    = (220, 30, 30) if on_rope else (180, 100, 20)
        alpha    = 230 if on_rope else 160

        surf = pygame.Surface((90, 90), pygame.SRCALPHA)
        c, r = 45, 45
        # 外圆
        pygame.draw.circle(surf, (*color, alpha // 2), (c, r), 34, 1)
        # 准星四线
        for dx, dy in [(-38, 0), (38, 0), (0, -38), (0, 38)]:
            pygame.draw.line(surf, (*color, alpha),
                             (c + dx // 3, r + dy // 3),
                             (c + dx, r + dy), 2)
        # X 标记 (剪刀刃)
        pygame.draw.line(surf, (*color, alpha), (c - 10, r - 10), (c + 10, r + 10), 3)
        pygame.draw.line(surf, (*color, alpha), (c + 10, r - 10), (c - 10, r + 10), 3)
        self.screen.blit(surf, (sx - c, sy - r))

        if on_rope:
            lbl = self.font_tip.render("⚡ 剪断!", True, (255, 80, 80))
            self.screen.blit(lbl, (sx + 38, sy - 14))

    def _draw_cut_effect(self):
        if self.state != STATE_CUTTING:
            return
        t     = self.cut_flash / self.CUT_FLASH_FRAMES
        alpha = int(t * 140)
        # 血红闪光
        flash = pygame.Surface((self.GAME_W, self.SCREEN_H), pygame.SRCALPHA)
        flash.fill((180, 10, 10, alpha))
        self.screen.blit(flash, (0, 0))
        # 文字
        text = self.font_lg.render("SNAP", True,
                                   (255, int(lerp(20, 100, 1 - t)), 20))
        scale = lerp(0.6, 1.3, 1 - t)
        w, h  = text.get_size()
        scaled = pygame.transform.scale(text, (int(w * scale), int(h * scale)))
        self.screen.blit(scaled,
                         (self.GAME_W // 2 - scaled.get_width() // 2,
                          self.SCREEN_H // 2 - scaled.get_height() // 2))

    def _draw_falling_scream(self):
        if self.state not in (STATE_FALLING, STATE_RESET):
            return
        # 速度线
        speed = min(abs(self.char_vy), 20)
        for _ in range(int(speed * 2)):
            lx = random.randint(0, self.GAME_W)
            ly = random.randint(0, self.SCREEN_H)
            length = random.randint(10, int(speed * 5))
            alpha  = random.randint(30, 100)
            ls = pygame.Surface((3, length), pygame.SRCALPHA)
            ls.fill((180, 80, 20, alpha))
            self.screen.blit(ls, (lx, ly))
        # 尖叫文字
        scream = self.font_md.render("AAAAAAH——", True, (200, 30, 30))
        self.screen.blit(scream,
                         (self.GAME_W // 2 - scream.get_width() // 2, 75))

    def _draw_vignette(self):
        """四角晕影 — 暗黑氛围感"""
        v = pygame.Surface((self.GAME_W, self.SCREEN_H), pygame.SRCALPHA)
        for r in range(0, 200, 8):
            alpha = int((r / 200) ** 2 * 160)
            pygame.draw.rect(v, (0, 0, 0, alpha),
                             (r, r, self.GAME_W - r * 2, self.SCREEN_H - r * 2), 8)
        self.screen.blit(v, (0, 0))

    def _draw_torch_glow(self):
        """火把光晕 — 从下方暖光照射"""
        flicker = math.sin(self.torch_phase) * 0.12 + math.sin(self.torch_phase * 2.7) * 0.06
        base_alpha = int((0.22 + flicker) * 255)
        radius = 320
        glow = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        for r in range(radius, 0, -12):
            a = int(base_alpha * (1 - r / radius) ** 1.6)
            a = max(0, min(255, a))
            pygame.draw.circle(glow, (255, 140, 30, a), (radius, radius), r)
        gx = self.ROPE_ATTACH_X - radius
        gy = self.ROPE_ATTACH_Y + self.ROPE_LENGTH - radius + 80
        self.screen.blit(glow, (gx, gy), special_flags=pygame.BLEND_RGBA_ADD)

    def _draw_hint(self):
        if self.state == STATE_HANGING:
            hint = self.font_tip.render(
                "✌ scissors gesture → aim at rope  |  SPACE=test  R=reset",
                True, GREY_DIM)
            self.screen.blit(hint, (12, self.SCREEN_H - 36))
        elif self.state == STATE_RESET:
            msg = self.font_md.render("Resetting...", True, GREY_DIM)
            self.screen.blit(msg,
                             (self.GAME_W // 2 - msg.get_width() // 2,
                              self.SCREEN_H // 2))

    def _draw_camera_panel(self, frame):
        panel_x = self.GAME_W
        panel_w = self.CAM_W
        panel_h = self.SCREEN_H

        # 暗色背景
        pygame.draw.rect(self.screen, (10, 8, 14), (panel_x, 0, panel_w, panel_h))

        if frame is not None:
            target_w = panel_w - 24
            target_h = int(target_w * frame.shape[0] / frame.shape[1])
            cam_resized = cv2.resize(frame, (target_w, target_h))
            cam_rgb     = cv2.cvtColor(cam_resized, cv2.COLOR_BGR2RGB)
            cam_surf    = pygame.surfarray.make_surface(
                np.transpose(cam_rgb, (1, 0, 2))
            )
            y_off = (panel_h - target_h) // 2
            self.screen.blit(cam_surf, (panel_x + 12, y_off))
            # 铁框边框
            pygame.draw.rect(self.screen, STONE_MID,
                             (panel_x + 12, y_off, target_w, target_h), 1)
            pygame.draw.rect(self.screen, IRON_COL,
                             (panel_x + 10, y_off - 2, target_w + 4, target_h + 4), 2)

        # 状态标签
        if self.scissors_active:
            st = self.font_sm.render("✂  DETECTED", True, (220, 80, 80))
        else:
            st = self.font_tip.render("show ✌  scissors gesture", True, GREY_DIM)
        self.screen.blit(st, (panel_x + panel_w // 2 - st.get_width() // 2, panel_h - 38))

        # 分隔线 (石墙缝)
        pygame.draw.line(self.screen, STONE_MID, (panel_x, 0), (panel_x, panel_h), 3)
        pygame.draw.line(self.screen, (5, 4, 8),  (panel_x + 1, 0), (panel_x + 1, panel_h), 1)

    def draw(self, frame):
        # 背景 (石砖地牢)
        if self._bg_surface is None:
            self._build_bg()
        self.screen.blit(self._bg_surface, (0, 0))

        # 火把暖光晕 (在角色和绳子之下)
        self._draw_torch_glow()

        # 绳子
        self._draw_rope()

        # 角色
        falling = self.state in (STATE_FALLING, STATE_RESET)
        if self.state == STATE_HANGING:
            sw = math.sin(self.swing_angle) * self.SWING_AMPLITUDE
            cx = self.ROPE_ATTACH_X + sw
            cy = self.ROPE_ATTACH_Y + self.ROPE_LENGTH
        else:
            cx, cy = self.char_x, self.char_y
        self._draw_character(cx, cy, falling=falling)

        # 四角晕影
        self._draw_vignette()

        # 剪刀准星
        self._draw_scissors_cursor()

        # 特效层
        self._draw_cut_effect()
        self._draw_falling_scream()
        self._draw_hint()

        # 摄像头面板
        self._draw_camera_panel(frame)

        pygame.display.flip()

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------
    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        self._reset_state()
                    elif event.key == pygame.K_SPACE:
                        if self.state == STATE_HANGING:
                            mid_y = self.ROPE_ATTACH_Y + self.ROPE_LENGTH * 0.5
                            self._trigger_cut(mid_y)

            frame = self._process_camera()
            self.update()
            self.draw(frame)
            self.clock.tick(60)

        self._cleanup()

    def _cleanup(self):
        self.cap.release()
        self.hand_landmarker.close()
        pygame.quit()
        sys.exit()


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    game = RopeCutGame()
    game.run()
