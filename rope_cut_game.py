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
# 颜色
# ---------------------------------------------------------------------------
SKY_TOP    = (100, 160, 220)
SKY_BOT    = (200, 230, 255)
GROUND_COL = (80,  160,  60)
BEAM_COL   = (90,   55,  20)
ROPE_COL   = (160,  100,  30)
SKIN_COL   = (255,  200, 150)
CLOTH_COL  = (60,   100, 200)
DARK_COL   = (40,    40,  40)
WHITE      = (255,  255, 255)
RED        = (220,   50,  50)
YELLOW     = (255,  220,   0)
GREEN      = (50,   220,  80)


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
        self.swing_angle     = 0.0          # 摆动相位
        self.char_x          = float(self.ROPE_ATTACH_X)
        self.char_y          = float(self.ROPE_ATTACH_Y + self.ROPE_LENGTH)
        self.char_vx         = 0.0
        self.char_vy         = 0.0
        self.cut_y           = 0.0          # 剪断位置 y
        self.cut_flash       = 0            # 特效剩余帧
        self.reset_timer     = 0            # 重置倒计时帧
        self.scissors_pos    = None         # (x, y) 游戏坐标
        self.scissors_active = False        # 当前帧剪刀手势是否检测到
        self.frays           = []           # 断口毛刺随机偏移列表

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
        # 生成断口毛刺
        self.frays = [
            (random.randint(-14, 14), random.randint(2, 18))
            for _ in range(8)
        ]

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

    # ------------------------------------------------------------------
    # 渲染
    # ------------------------------------------------------------------
    def _build_bg(self):
        surf = pygame.Surface((self.GAME_W, self.SCREEN_H))
        draw_gradient_rect(surf, SKY_TOP, SKY_BOT, (0, 0, self.GAME_W, self.SCREEN_H))
        # 地面
        pygame.draw.rect(surf, GROUND_COL,
                         (0, self.SCREEN_H - 65, self.GAME_W, 65))
        pygame.draw.rect(surf, (60, 130, 40),
                         (0, self.SCREEN_H - 65, self.GAME_W, 8))
        # 横梁
        pygame.draw.rect(surf, BEAM_COL, (0, 0, self.GAME_W, 52))
        pygame.draw.rect(surf, (110, 70, 25), (0, 48, self.GAME_W, 6))
        # 竖梁
        pygame.draw.rect(surf, BEAM_COL,
                         (self.ROPE_ATTACH_X - 18, 0, 36, 80))
        self._bg_surface = surf

    def _draw_rope(self):
        pts = []
        if self.state == STATE_HANGING:
            n = 24
            for i in range(n + 1):
                t = i / n
                y = self.ROPE_ATTACH_Y + t * self.ROPE_LENGTH
                x = self._rope_x_at_y(y)
                pts.append((int(x), int(y)))
            pygame.draw.lines(self.screen, ROPE_COL, False, pts, 5)

        elif self.state in (STATE_CUTTING, STATE_FALLING, STATE_RESET):
            # 上半段绳子 (仍挂着)
            t_cut = (self.cut_y - self.ROPE_ATTACH_Y) / max(self.ROPE_LENGTH, 1)
            t_cut = max(0.05, min(0.95, t_cut))
            n = 12
            pts_top = []
            for i in range(n + 1):
                t = i / n * t_cut
                y = self.ROPE_ATTACH_Y + t * self.ROPE_LENGTH
                x = self._rope_x_at_y(y)
                pts_top.append((int(x), int(y)))
            if len(pts_top) > 1:
                pygame.draw.lines(self.screen, ROPE_COL, False, pts_top, 5)

            # 断口毛刺
            cx = int(self._rope_x_at_y(self.cut_y))
            cy = int(self.cut_y)
            for fx, fy in self.frays:
                pygame.draw.line(self.screen, ROPE_COL,
                                 (cx, cy), (cx + fx, cy + fy), 2)

    def _draw_character(self, x, y, falling=False):
        ix, iy = int(x), int(y)
        angle = 0.0
        if falling:
            # 坠落时身体旋转
            angle = min(self.char_vy * 3.5, 90)

        # --- 绘制到临时 surface 再旋转 ---
        size = 160
        tmp = pygame.Surface((size, size), pygame.SRCALPHA)
        cx, cy_base = size // 2, size // 2 - 20

        # 身体
        pygame.draw.line(tmp, DARK_COL,
                         (cx, cy_base + 22), (cx, cy_base + 75), 5)
        # 衣服
        pygame.draw.polygon(tmp, CLOTH_COL, [
            (cx - 14, cy_base + 28),
            (cx + 14, cy_base + 28),
            (cx + 16, cy_base + 65),
            (cx - 16, cy_base + 65),
        ])
        # 手臂
        arm_swing = math.sin(self.swing_angle * 2) * 12 if not falling else 35
        pygame.draw.line(tmp, SKIN_COL,
                         (cx, cy_base + 35),
                         (cx - 38, cy_base + 50 + int(arm_swing)), 5)
        pygame.draw.line(tmp, SKIN_COL,
                         (cx, cy_base + 35),
                         (cx + 38, cy_base + 50 - int(arm_swing)), 5)
        # 腿
        leg_swing = math.sin(self.swing_angle * 2) * 10 if not falling else 25
        pygame.draw.line(tmp, CLOTH_COL,
                         (cx, cy_base + 75),
                         (cx - 22, cy_base + 120 + int(leg_swing)), 6)
        pygame.draw.line(tmp, CLOTH_COL,
                         (cx, cy_base + 75),
                         (cx + 22, cy_base + 120 - int(leg_swing)), 6)
        # 头
        pygame.draw.circle(tmp, SKIN_COL, (cx, cy_base), 22)
        # 眼睛
        if falling:
            # 惊讶表情
            pygame.draw.circle(tmp, DARK_COL, (cx - 8, cy_base - 2), 5)
            pygame.draw.circle(tmp, DARK_COL, (cx + 8, cy_base - 2), 5)
            pygame.draw.arc(tmp, DARK_COL,
                            pygame.Rect(cx - 8, cy_base + 6, 16, 10),
                            0, math.pi, 2)
        else:
            pygame.draw.circle(tmp, DARK_COL, (cx - 7, cy_base - 3), 4)
            pygame.draw.circle(tmp, DARK_COL, (cx + 7, cy_base - 3), 4)
            pygame.draw.arc(tmp, DARK_COL,
                            pygame.Rect(cx - 7, cy_base + 5, 14, 8),
                            math.pi, 2 * math.pi, 2)
        # 头发
        pygame.draw.arc(tmp, DARK_COL,
                        pygame.Rect(cx - 22, cy_base - 22, 44, 44),
                        0, math.pi, 5)

        # 旋转并贴图
        rotated = pygame.transform.rotate(tmp, -angle)
        rw, rh = rotated.get_size()
        self.screen.blit(rotated, (ix - rw // 2, iy - 20 - rh // 4))

    def _draw_scissors_cursor(self):
        if not self.scissors_active or not self.scissors_pos:
            return
        sx, sy = int(self.scissors_pos[0]), int(self.scissors_pos[1])
        on_rope = self._scissors_near_rope()
        color   = RED if on_rope else YELLOW
        alpha   = 200 if on_rope else 140

        # 准星
        surf = pygame.Surface((80, 80), pygame.SRCALPHA)
        cx, cy = 40, 40
        pygame.draw.line(surf, (*color, alpha), (cx - 28, cy), (cx + 28, cy), 3)
        pygame.draw.line(surf, (*color, alpha), (cx, cy - 28), (cx, cy + 28), 3)
        pygame.draw.circle(surf, (*color, alpha // 2), (cx, cy), 26, 2)
        # 剪刀图标 (简化)
        pygame.draw.line(surf, (*color, alpha), (cx - 8, cy - 8), (cx + 10, cy + 10), 3)
        pygame.draw.line(surf, (*color, alpha), (cx + 8, cy - 8), (cx - 10, cy + 10), 3)
        self.screen.blit(surf, (sx - 40, sy - 40))

        if on_rope:
            label = self.font_tip.render("对准绳子! 剪!",  True, RED)
            self.screen.blit(label, (sx + 30, sy - 12))

    def _draw_cut_effect(self):
        if self.state != STATE_CUTTING:
            return
        t = self.cut_flash / self.CUT_FLASH_FRAMES
        alpha = int(t * 180)
        flash = pygame.Surface((self.GAME_W, self.SCREEN_H), pygame.SRCALPHA)
        flash.fill((255, 240, 0, alpha))
        self.screen.blit(flash, (0, 0))

        size = int(lerp(30, 90, 1 - t))
        text = self.font_lg.render("✂ 咔嚓!", True,
                                   (int(lerp(220, 255, t)),
                                    int(lerp(50, 200, t)), 0))
        self.screen.blit(text,
                         (self.GAME_W // 2 - text.get_width() // 2,
                          self.SCREEN_H // 2 - text.get_height() // 2 - size))

    def _draw_falling_scream(self):
        if self.state not in (STATE_FALLING, STATE_RESET):
            return
        scream = self.font_md.render("啊啊啊——!!!", True, RED)
        self.screen.blit(scream,
                         (self.GAME_W // 2 - scream.get_width() // 2, 80))

    def _draw_hint(self):
        if self.state == STATE_HANGING:
            hint = self.font_sm.render(
                "对摄像头做✌ 手势  对准绳子即可剪断  |  空格=手动触发  R=重置",
                True, (50, 50, 80))
            self.screen.blit(hint, (12, self.SCREEN_H - 44))

        elif self.state == STATE_RESET:
            msg = self.font_md.render("即将重置...", True, (80, 80, 80))
            self.screen.blit(msg,
                             (self.GAME_W // 2 - msg.get_width() // 2,
                              self.SCREEN_H // 2))

    def _draw_camera_panel(self, frame):
        """在右侧绘制摄像头画面"""
        panel_x = self.GAME_W
        panel_w = self.CAM_W
        panel_h = self.SCREEN_H

        # 背景
        pygame.draw.rect(self.screen, (20, 20, 30),
                         (panel_x, 0, panel_w, panel_h))

        if frame is not None:
            # 缩放到合适尺寸
            target_w = panel_w - 20
            target_h = int(target_w * frame.shape[0] / frame.shape[1])
            cam_resized = cv2.resize(frame, (target_w, target_h))
            cam_rgb = cv2.cvtColor(cam_resized, cv2.COLOR_BGR2RGB)
            cam_surf = pygame.surfarray.make_surface(
                np.transpose(cam_rgb, (1, 0, 2))
            )
            y_off = (panel_h - target_h) // 2
            self.screen.blit(cam_surf, (panel_x + 10, y_off))
            pygame.draw.rect(self.screen, WHITE,
                             (panel_x + 10, y_off, target_w, target_h), 2)

        # 状态标签
        if self.scissors_active:
            status_text = self.font_sm.render("✂  剪刀手势 检测到!", True, GREEN)
        else:
            status_text = self.font_sm.render("请做✌ 剪刀手势", True, (180, 180, 180))
        self.screen.blit(status_text,
                         (panel_x + panel_w // 2 - status_text.get_width() // 2,
                          panel_h - 42))

        # 分隔线
        pygame.draw.line(self.screen, (80, 80, 80),
                         (panel_x, 0), (panel_x, panel_h), 2)

        # 标题
        title = self.font_tip.render("摄像头  Camera", True, (150, 150, 170))
        self.screen.blit(title, (panel_x + panel_w // 2 - title.get_width() // 2, 8))

    def draw(self, frame):
        # 背景
        if self._bg_surface is None:
            self._build_bg()
        self.screen.blit(self._bg_surface, (0, 0))

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

        # 剪刀位置指示
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
