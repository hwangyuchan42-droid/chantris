
import random
import numpy as np
import math

# --- 기본 설정
width = 780
height = 610
fps = 60

white = (255, 255, 255)
cyan = (0, 200, 160)
yellow = (255, 255, 0)
mazenta = (255, 0, 255)
orange = (255, 127, 0)
blue = (0, 0, 255)
green = (0, 255, 0)
red = (255, 0, 0)
gray = (127, 127, 127)
lightgray = (200, 200, 200)
black = (0, 0, 0)
cell_Colors = [white, cyan, yellow, mazenta, orange, blue, green, red, gray, black]

# 변경: 고정값들을 1.2배 (반올림 적용). 원래값: DAS=10, ARR=5, SD_ARR=1 => 1.2배 -> DAS=12, ARR=6, SD_ARR stays 1
pygame.init()
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("유찬 tris")
clock = pygame.time.Clock()
mFont = pygame.font.SysFont("arial", 50, True, False)
sFont = pygame.font.SysFont("arial", 22, False, False)

# --- 무지개 텍스트 출력 함수
def draw_rainbow_text(text, font, x, y):
    rainbow_colors = [red, orange, yellow, green, blue, mazenta]
    offset_x = x
    for i, ch in enumerate(text):
        color = rainbow_colors[i % len(rainbow_colors)]
        ch_surface = font.render(ch, True, color)
        screen.blit(ch_surface, (offset_x, y))
        offset_x += ch_surface.get_width()

sText = mFont.render("chantris", True, white)
pressText = sFont.render("Press Enter to Start", True, black)
press_rect = pressText.get_rect()
press_rect.centerx = round(width / 2)
press_rect.centery = round(height / 2) + 120

# --- 통계 및 표시용 전역 변수 추가
start_time = pygame.time.get_ticks()
total_clears = 0
last_clear_time = 0
last_clear_text = ""
score = 0
placed_pieces = 0
run = True
GameStart = False
game_over = False

# game over fade 관리
game_over_since = None
game_over_fade_ms = 2000  # 2초 동안 페이드인

# 기본 중력(밀리초 간격) — 점점 빨라짐(레벨 기준)
gravity_interval_ms = 500
lock_delay_ms = 500
gravity_timer = 0
lock_timer = 0

is_touching = False
hold = np.full((6, 2), -1, int)

# --- 마지막 잠긴 미노 정보 (T-Spin 판정용)
last_locked_mino_type = -1
last_locked_blocks = []
last_locked_pivot = None
last_locked_was_tspin = False

# --- 별 반짝임 효과용 (시간 기반 오프셋 포함)
stars = []
for _ in range(120):
    x = random.randint(0, width)
    y = random.randint(0, height)
    twinkle_speed = random.uniform(0.005, 0.02)
    fall_speed = random.uniform(0.05, 0.5)  # 살짝 느리게
    base_phase = random.uniform(0, 2*math.pi)
    wobble = random.uniform(2, 12)  # x방향 이동 진폭
    stars.append([x, y, twinkle_speed, base_phase, fall_speed, wobble])

def draw_starry_background(now_time):
    screen.fill((10, 10, 30))
    t = now_time / 1000.0
    for s in stars:
        x, y, speed, phase, fall, wobble = s

        # phase 업데이트
        phase += speed
        if phase > 2 * math.pi:
            phase -= 2 * math.pi
        s[3] = phase

        # 밝기 계산
        brightness = int(155 + 100 * abs(0.5 - ((math.sin(phase) + 1) / 2)))

        # y 위치 아래로 이동 (fall 속도를 2~5배 정도 빠르게)
        y += fall * 3  # 기본보다 3배 빠르게
        if y > height:
            y = -2
            x = random.randint(0, width)
            fall = random.uniform(0.2, 1.0)  # 초기 속도도 빠르게
            s[4] = fall

        # 좌표 업데이트
        s[0], s[1] = x, y

        # 화면에 그릴 때만 x 흔들림 적용
        x_draw = int(x + math.sin(t * 0.5 + phase) * wobble) % width  # 흔들림도 조금 빠르게
        pygame.draw.circle(screen, (brightness, brightness, brightness), (x_draw, int(y)), 1)

# --- 마지막 액션 기록 (rotate이면 'rotate')
last_action = None

# --- 파티클 클래스
class Particle:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.vx = random.uniform(-100, 100)
        self.vy = random.uniform(-250, -50)
        self.life = random.uniform(0.4, 0.9)
        self.age = 0.0
        self.color = color
        self.size = random.randint(3, 7)

    def update(self, dt):
        self.age += dt
        self.vy += 600 * dt
        self.x += self.vx * dt
        self.y += self.vy * dt

    def draw(self):
        alpha = max(0, 1 - self.age / self.life)
        if alpha <= 0: return
        surf = pygame.Surface((self.size, self.size), pygame.SRCALPHA)
        col = (*self.color, int(255 * alpha))
        surf.fill(col)
        screen.blit(surf, (int(self.x - self.size/2), int(self.y - self.size/2)))

particles = []

class inputs:
    moveLeft = False
    moveRight = False
    softDrop = False
    hardDrop = False
    rotateLeft = False
    rotateRight = False
    hold = False
    DAS_LEFT = False
    DAS_RIGHT = False
    SD_ARR_CNT = 0
    L_DAS_CNT = 0
    L_ARR_CNT = 0
    R_DAS_CNT = 0
    R_ARR_CNT = 0
    SD_ARR_VALUE = 1  
    DAS_VALUE = 10
    ARR_VALUE = 4

class field:
    matrix = np.zeros((40, 10), int)
    testMatrix = np.zeros((40, 10), int)
    def __init__(self):
        self.clearCnt = 0
        self.tspin = False

    def clearLines(self):
        global total_clears, last_clear_time, last_clear_text, score, last_locked_was_tspin, combo

        lines_cleared = 0 
        cleared_cells = []
        # 보이는 20..39행 인덱스(화면)
        for row_idx in range(20, 40):
            if np.all(self.testMatrix[row_idx]):
                for col in range(10):
                    color_idx = self.testMatrix[row_idx, col]
                    if color_idx != 0:
                        cleared_cells.append((row_idx, col, int(color_idx)))
                # 한 줄 제거
                self.testMatrix[1:row_idx+1] = self.testMatrix[0:row_idx]
                self.matrix[1:row_idx+1] = self.matrix[0:row_idx]
                self.testMatrix[0] = 0
                self.matrix[0] = 0
                self.clearCnt += 1
                lines_cleared += 1
        if len(cleared_cells) > 0:
            spawn_particles_from_cells(cleared_cells)

        # T-spin 판정은 락 직전에 세팅된 last_locked_was_tspin 사용
        tspin_detected = False
        if lines_cleared > 0 and last_locked_was_tspin:
            tspin_detected = True

        # 점수/텍스트 처리(원본 로직 유지)
        if lines_cleared > 0:
            total_clears += lines_cleared
            last_clear_time = pygame.time.get_ticks()

            if tspin_detected:
                if lines_cleared == 1:
                    score_inc = 1000
                    text1 = ("T-SPIN", mazenta)
                    text2 = ("SINGLE", white)
                elif lines_cleared == 2:
                    score_inc = 2000
                    text1 = ("T-SPIN", mazenta)
                    text2 = ("DOUBLE", white)
                elif lines_cleared == 3:
                    score_inc = 3000
                    text1 = ("T-SPIN", mazenta)
                    text2 = ("TRIPLE", white)
                else:
                    score_inc = lines_cleared * 100
                    text1 = (f"T-SPIN x{lines_cleared}", mazenta)
                    text2 = None
                score += score_inc + lines_cleared * 100
                last_clear_text = (text1, text2)
            else:
                if lines_cleared == 1:
                    score += 50
                    last_clear_text = (("Single", white), None)
                elif lines_cleared == 2:
                    score += 500
                    last_clear_text = (("Double", white), None)
                elif lines_cleared == 3:
                    score += 1000
                    last_clear_text = (("Triple", white), None)
                else:
                    score += 2000
                    last_clear_text = (("Quad", white), None)

            # combo 처리
            global last_combo_text
            if 'combo' not in globals():
                combo = -1
            combo += 1
            combo_bonus = combo * 50
            score += combo_bonus
            if combo >= 2:
                last_combo_text = [f"{combo} Combo!", (255, 215, 0)]
            else:
                last_combo_text = None
        else:
            combo = -1

        # ALL CLEAR 처리: 필드에 아무 블록도 없을 때
        if np.count_nonzero(self.testMatrix) == 0:
            score += 4000
            last_clear_time = pygame.time.get_ticks()
            last_clear_text = (("ALL", "rainbow"), ("CLEAR", "rainbow"))

        # clear 후 사용을 위해 last_locked_was_tspin 초기화는 호출자가 책임지도록 둘 수도 있지만
        # 원래 코드 흐름을 유지하기 위해 리턴 후 호출자가 계속 진행하도록 한다.
        return lines_cleared, tspin_detected

# --- mino/ bag 등 기존 클래스 (원본 로직 유지)
class mino:
    I = np.array([[0,1],[1,1],[2,1],[3,1],[1,0],[3,19]])
    O = np.array([[0,0],[0,1],[1,0],[1,1],[2,0],[4,19]])
    T = np.array([[0,1],[1,0],[1,1],[2,1],[3,0],[3,19]])
    L = np.array([[0,1],[1,1],[2,0],[2,1],[4,0],[3,19]])
    J = np.array([[0,0],[0,1],[1,1],[2,1],[5,0],[3,19]])
    S = np.array([[0,1],[1,0],[1,1],[2,0],[6,0],[3,19]])
    Z = np.array([[0,0],[1,0],[1,1],[2,1],[7,0],[3,19]])
    X = np.zeros((6,2), int)
    minoData = [I,O,T,L,J,S,Z]
    rotMat = np.array([[[1,0],[0,1]], [[0,-1],[1,0]], [[-1,0],[0,-1]], [[0,1],[-1,0]]])
    SRS2 = np.array([[[0,0],[0,0],[0,0],[0,0],[0,0]]] * 4)
    SRS3 = np.array([[[0,0],[0,0],[0,0],[0,0],[0,0]],
                     [[0,0],[1,0],[1,1],[0,-2],[1,-2]],
                     [[0,0],[0,0],[0,0],[0,0],[0,0]],
                     [[0,0],[-1,0],[-1,1],[0,-2],[-1,-2]]])
    SRS4 = np.array([[[0,0],[-1,0],[2,0],[-1,0],[2,0]],
                     [[0,0],[1,0],[1,0],[1,-1],[1,2]],
                     [[0,0],[2,0],[-1,0],[2,1],[-1,1]],
                     [[0,0],[0,0],[0,0],[0,2],[0,-1]]])
    SRS = np.array([SRS2, SRS3, SRS4])
    nexts = [0,0,0,0,0]

    def __init__(self, newMino):
        self.data = newMino.copy()
        self.testdata = self.data.copy()
        self.ghost = self.data.copy()

    def __del__(self):
        if self.data[4,1] != 0:
            for i in range(self.data[4,1]):
                self.rotateMino(-1)

    def drawMino(self):
        for i in range(4):
            y = self.data[i,1] + self.data[5,1]
            x = self.data[i,0] + self.data[5,0]
            if 0 <= y < field.matrix.shape[0] and 0 <= x < field.matrix.shape[1]:
                field.matrix[y,x] = self.data[4,0]

    def eraseMino(self):
        for i in range(4):
            y = self.data[i,1] + self.data[5,1]
            x = self.data[i,0] + self.data[5,0]
            if 0 <= y < field.matrix.shape[0] and 0 <= x < field.matrix.shape[1]:
                field.matrix[y,x] = 0

    def isBlockedByMovement(self, toX, toY):
        x, y = toX + self.data[5,0], toY + self.data[5,1]
        for i in range(4):
            nx = self.data[i,0] + x
            ny = self.data[i,1] + y
            if not ((nx) in range(10) and (ny) in range(40) and field.testMatrix[ny,nx] == 0):
                return False
        return True

    def moveMino(self, toX, toY):
        self.data[5,0] += toX
        self.data[5,1] += toY

    def isSRS(self, d):
        self.eraseMino()
        self.testdata = self.data.copy()
        ad = self.data[4,1].copy()
        self.rotateMino(d)
        bd = self.data[4,1].copy()
        for i in range(5):
            idx = np.max(self.minoData[self.data[4,0]-1][:4]) - 1
            dx = self.SRS[idx, ad, i, 0] - self.SRS[idx, bd, i, 0]
            dy = self.SRS[idx, ad, i, 1] - self.SRS[idx, bd, i, 1]
            if self.isBlockedByMovement(dx, dy):
                self.moveMino(dx, dy)
                self.data[:5] = self.testdata[:5].copy()
                return True
        self.data[:5] = self.testdata[:5].copy()
        self.drawMino()
        return False

    def rotateMino(self, d):
        self.data[4,1] = (self.data[4,1] + d + 4) % 4
        for i in range(4):
            base_x = self.data[i,0] + np.max(self.minoData[self.data[4,0]-1][:4]) * np.min(self.rotMat[d,1])
            base_y = self.data[i,1] + np.max(self.minoData[self.data[4,0]-1][:4]) * np.min(self.rotMat[d,0])
            self.data[i] = np.dot(self.rotMat[d], [base_x, base_y])

    def drawGhost(self):
        self.testdata = self.data.copy()
        while (self.isBlockedByMovement(0,1)):
            self.moveMino(0,1)
        self.ghost = self.data.copy()
        self.data = self.testdata.copy()

    def hardDrop(self):
        self.eraseMino()
        self.data = self.ghost.copy()
        self.drawMino()

class bag:
    def __init__(self):
        self.nowQueue = random.sample(mino.minoData, 7) + random.sample(mino.minoData, 7)
    def generateBag(self):
        if len(self.nowQueue) < 10:
            self.nowQueue += random.sample(mino.minoData, 7)

# --- 필드/백업 객체 생성
f = field()
nowBag = bag()
nowMino = mino(nowBag.nowQueue.pop(0))
nowMino.drawMino()

# 보조: 스폰 위치로 미노 초기화
def spawn_new_mino_from_template(template):
    global game_over
    m = mino(template)
    m.data[5,0] = int(template[5,0].copy())
    m.data[5,1] = int(template[5,1].copy())
    m.data[4,1] = 0
    for i in range(4):
        y = m.data[i,1] + m.data[5,1]
        x = m.data[i,0] + m.data[5,0]
        if 0 <= y < field.testMatrix.shape[0] and 0 <= x < field.testMatrix.shape[1]:
            if field.testMatrix[y,x] != 0 or field.matrix[y,x] != 0:
                game_over = True
                return m
    return m

def reset_game():
    global f, nowBag, nowMino, hold, gravity_timer, lock_timer, is_touching, inputs, game_over, GameStart
    global start_time, total_clears, last_clear_time, last_clear_text, score, placed_pieces, particles
    global last_locked_mino_type, last_locked_blocks, last_locked_pivot, last_locked_was_tspin, last_action
    global next_attack_time, attack_pending, attack_start_time, attack_ready, attack_lines
    global game_over_since

    f = field()
    nowBag = bag()
    nowMino = spawn_new_mino_from_template(nowBag.nowQueue.pop(0))
    field.matrix = np.zeros((40,10), int)
    field.testMatrix = np.zeros((40,10), int)
    nowMino.drawMino()
    hold = np.full((6,2), -1, int)
    gravity_timer = 0
    lock_timer = 0
    is_touching = False
    inputs.SD_ARR_CNT = inputs.L_DAS_CNT = inputs.L_ARR_CNT = inputs.R_DAS_CNT = inputs.R_ARR_CNT = 0
    inputs.moveLeft = inputs.moveRight = inputs.softDrop = False
    inputs.hold = False
    game_over = False
    GameStart = True
    start_time = pygame.time.get_ticks()
    total_clears = 0
    last_clear_time = 0
    last_clear_text = ""
    score = 0
    placed_pieces = 0
    particles = []
    last_locked_mino_type = -1
    last_locked_blocks = []
    last_locked_pivot = None
    last_locked_was_tspin = False
    last_action = None

    # 공격 초기화: 15~30초로 난이도 적당히 느리게
    next_attack_time = pygame.time.get_ticks() + random.randint(15000, 30000)
    attack_pending = False
    attack_start_time = 0
    attack_ready = False
    attack_lines = 0

    game_over_since = None

# 파티클 스폰 헬퍼
CELL_SIZE = 30
FIELD_X = 241
FIELD_Y = 1
VISIBLE_TOP = 20

def spawn_particles_from_cells(cells):
    for (r,c,color_idx) in cells:
        px = c * CELL_SIZE + FIELD_X + CELL_SIZE/2
        py = (r - VISIBLE_TOP) * CELL_SIZE + FIELD_Y + CELL_SIZE/2
        for _ in range(random.randint(6,10)):
            color = cell_Colors[color_idx] if 0 <= color_idx < len(cell_Colors) else gray
            particles.append(Particle(px + random.uniform(-6,6), py + random.uniform(-6,6), color))

# --- 공격 적용 함수 (즉시 하단에 garbage 추가, 위로 밀림)
def apply_attack(lines):
    global field, game_over
    if lines <= 0:
        return
    lines = min(lines, 40)
    # 위로 shift
    if lines >= 40:
        field.matrix[:] = 0
        field.testMatrix[:] = 0
    else:
        field.matrix[:-lines] = field.matrix[lines:]
        field.testMatrix[:-lines] = field.testMatrix[lines:]
        field.matrix[:lines] = 0
        field.testMatrix[:lines] = 0
    # 하단에 garbage 채우기
    for i in range(40 - lines, 40):
        hole = random.randint(0, 9)
        new_line = np.ones(10, int) * 8
        new_line[hole] = 0
        field.matrix[i] = new_line.copy()
        field.testMatrix[i] = new_line.copy()

# 공격 변수 초기값
next_attack_time = pygame.time.get_ticks() + random.randint(15000, 30000)
attack_pending = False
attack_start_time = 0
attack_ready = False
attack_lines = 0

# --- 메인 루프
while run:
    dt_ms = clock.tick(fps)
    dt = dt_ms / 1000.0
    now_time = pygame.time.get_ticks()

    # --- 중력 속도(레벨 기반으로 점점 빨라짐)
    # 레벨 = total_clears // 10, 매 레벨당 30ms 감소, 최저 60ms
    level = total_clears // 10
    gravity_interval_ms = max(60, 500 - level * 30)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                if not GameStart or game_over:
                    # Enter: 게임 오버 상태면 재시작, 메인 화면이면 시작
                    reset_game()
                    continue
                else:
                    GameStart = True

            # 메인/오버 화면에서 ESC로 메인화면 돌아가기
            if event.key == pygame.K_ESCAPE:
                if game_over:
                    # 오버->메인
                    game_over = False
                    GameStart = False
                    game_over_since = None
                    continue

            if event.key == pygame.K_p and GameStart and not game_over:
                nowMino.eraseMino()
                field.matrix[:-1] = field.matrix[1:]
                field.testMatrix[:-1] = field.testMatrix[1:]
                hole = random.randint(0,9)
                new_line = np.ones(10,int)*8
                new_line[hole] = 0
                field.matrix[-1] = new_line.copy()
                field.testMatrix[-1] = new_line.copy()
                nowMino.drawMino()
                for i_block in range(4):
                    yb = nowMino.data[i_block,1] + nowMino.data[5,1]
                    xb = nowMino.data[i_block,0] + nowMino.data[5,0]
                    if 0 <= yb < field.testMatrix.shape[0] and 0 <= xb < field.testMatrix.shape[1]:
                        if field.testMatrix[yb,xb] != 0:
                            game_over = True
                            GameStart = False
                            game_over_since = now_time
                            break

            if not GameStart or game_over:
                continue

            # 하드드랍
            if event.key == pygame.K_SPACE:
                nowMino.eraseMino()
                nowMino.data = nowMino.ghost.copy()
                last_locked_blocks = []
                for i in range(4):
                    y = nowMino.data[i,1] + nowMino.data[5,1]
                    x = nowMino.data[i,0] + nowMino.data[5,0]
                    last_locked_blocks.append((int(y), int(x)))
                last_locked_mino_type = int(nowMino.data[4,0])
                if last_locked_mino_type == 3:
                    last_locked_pivot = (int(nowMino.data[2,1] + nowMino.data[5,1]), int(nowMino.data[2,0] + nowMino.data[5,0]))
                else:
                    last_locked_pivot = None

                field.testMatrix = field.matrix.copy()
                for (r,c) in last_locked_blocks:
                    if 0 <= r < field.testMatrix.shape[0] and 0 <= c < field.testMatrix.shape[1]:
                        field.testMatrix[r,c] = last_locked_mino_type

                last_locked_was_tspin = False
                if last_locked_mino_type == 3 and last_action == 'rotate' and last_locked_pivot is not None:
                    pr, pc = last_locked_pivot
                    corners = [(pr-1,pc-1),(pr-1,pc+1),(pr+1,pc-1),(pr+1,pc+1)]
                    occupied = 0
                    for (cr,cc) in corners:
                        if cr<0 or cr>=field.testMatrix.shape[0] or cc<0 or cc>=field.testMatrix.shape[1]:
                            occupied += 1
                        else:
                            if field.testMatrix[cr,cc] != 0:
                                occupied += 1
                    if occupied >= 3:
                        last_locked_was_tspin = True

                nowMino.drawMino()
                placed_pieces += 1

                # 라인 클리어 계산 (clearLines는 lines_cleared, tspin 반환)
                lines_cleared, tspin_detected = f.clearLines()

                # --- 상쇄 처리: 공격이 뜬 상태이면 지운만큼/TS핀은 2배로 깎음
                if attack_pending and lines_cleared > 0:
                    cancel_amount = lines_cleared * (2 if tspin_detected else 1)
                    attack_lines -= cancel_amount
                    if attack_lines <= 0:
                        # 공격 완전 상쇄: 취소 및 다음 공격 스케줄
                        attack_pending = False
                        attack_ready = False
                        attack_lines = 0
                        next_attack_time = now_time + random.randint(15000, 30000)

                # 하드드랍 후 공격이 준비(빨강)이면 적용(우선순위: 상쇄 먼저 처리했으므로 attack_lines >0이면 적용)
                if attack_pending and attack_ready and attack_lines > 0:
                    apply_attack(attack_lines)
                    attack_pending = False
                    attack_ready = False
                    next_attack_time = now_time + random.randint(15000, 30000)

                nowMino = spawn_new_mino_from_template(nowBag.nowQueue.pop(0))
                nowBag.generateBag()
                nowMino.drawMino()
                inputs.hardDrop = True
                inputs.hold = False
                lock_timer = 0
                is_touching = False
                last_action = None
                last_locked_was_tspin = False

            # 회전
            if event.key == pygame.K_UP and nowMino.isSRS(1):
                nowMino.rotateMino(1)
                nowMino.drawMino()
                lock_timer = 0
                last_action = 'rotate'
            if event.key == pygame.K_z and nowMino.isSRS(-1):
                nowMino.rotateMino(-1)
                nowMino.drawMino()
                lock_timer = 0
                last_action = 'rotate'
            if event.key == pygame.K_a and nowMino.isSRS(-2):
                nowMino.rotateMino(-2)
                nowMino.drawMino()
                lock_timer = 0
                last_action = 'rotate'

            # 홀드
            if event.key == pygame.K_c and inputs.hold == False:
                nowMino.eraseMino()
                if hold[4,0] == -1:
                    hold = nowMino.data.copy()
                    nowMino = spawn_new_mino_from_template(nowBag.nowQueue.pop(0))
                    nowBag.generateBag()
                else:
                    tmp = hold.copy()
                    hold = nowMino.data.copy()
                    nowMino = mino(tmp)
                    nowMino.data[5] = mino.minoData[nowMino.data[4,0]-1][5].copy()
                    nowMino.data[4,1] = 0
                    nowMino.drawMino()
                inputs.hold = True
                lock_timer = 0
                is_touching = False
                last_action = None

            # 좌우하
            if event.key == pygame.K_LEFT and nowMino.isBlockedByMovement(-1,0):
                nowMino.eraseMino()
                nowMino.moveMino(-1,0)
                nowMino.drawMino()
                inputs.moveLeft = True
                lock_timer = 0
                last_action = None
            if event.key == pygame.K_RIGHT and nowMino.isBlockedByMovement(1,0):
                nowMino.eraseMino()
                nowMino.moveMino(1,0)
                nowMino.drawMino()
                inputs.moveRight = True
                lock_timer = 0
                last_action = None
            if event.key == pygame.K_DOWN and nowMino.isBlockedByMovement(0,1):
                inputs.softDrop = True
                last_action = None

        if event.type == pygame.KEYUP:
            if event.key == pygame.K_LEFT:
                inputs.L_DAS_CNT = 0
                inputs.L_ARR_CNT = 0
                inputs.moveLeft = False
            if event.key == pygame.K_RIGHT:
                inputs.R_DAS_CNT = 0
                inputs.R_ARR_CNT = 0
                inputs.moveRight = False
            if event.key == pygame.K_DOWN:
                inputs.SD_ARR_CNT = 0
                inputs.softDrop = False

    # --- 공격 생성 타이밍 체크 ---
    if not attack_pending and now_time >= next_attack_time and GameStart and not game_over:
        attack_pending = True
        attack_start_time = now_time
        attack_ready = False
        # 공격량 계산: score//10000 + 2, 최소1 최대10
        attack_lines = score // 10000 + 2
        attack_lines = max(1, attack_lines)
        attack_lines = min(attack_lines, 10)

    if attack_pending and not attack_ready and now_time - attack_start_time >= 5000:
        attack_ready = True

    # 게임 미시작/오버시 메뉴 표시
    if not GameStart and not game_over:
        # 메인 화면 (애니메이션 포함)
        draw_starry_background(now_time)
        # 타이틀 중앙
        title_x = width // 2
        title_y = 80
        screen.blit(sText, (title_x - sText.get_width() // 2, title_y))

        # 타이틀 주변을 도는 미노(위성)
        tsec = now_time / 1000.0
        orbit_radii = [80, 110, 140]
        colors = [cyan, yellow, mazenta]
        for i in range(6):
            angle = tsec * 0.8 + (i * math.pi/3)
            r = orbit_radii[i % len(orbit_radii)]
            ox = int(title_x + math.cos(angle) * r)
            oy = int(title_y + 10 + math.sin(angle) * (r*0.35))
            # 작은 미노를 원형으로 표시 (간단한 블록)
            size = 14
            rect = pygame.Rect(ox - size//2, oy - size//2, size, size)
            pygame.draw.rect(screen, colors[i % len(colors)], rect)
            pygame.draw.rect(screen, black, rect, 2)

        # Press Enter 깜빡임
        blink = (now_time // 600) % 2 == 0
        if blink:
            screen.blit(pressText, press_rect)

        # 하단 설명
        lines = [
            "Controls:",
            "- Left / Right : Move",
            "- Up / Z / X : Rotate",
            "- Down : Soft Drop",
            "- Space : Hard Drop",
            "- C : Hold",
            "",
            "made my 30233 HwangYuChan. 2025"
        ]
        for i, l in enumerate(lines):
            txt = sFont.render(l, True, white)
            screen.blit(txt, (width // 2 - 220, 160 + i * 28))

        pygame.display.flip()
        continue

    if not GameStart and game_over:
        # 게임 오버 화면: 페이드인 효과
        draw_starry_background(now_time)
        # 준비: 설정된 game_over_since가 없으면 지금 설정
        if game_over_since is None:
            game_over_since = now_time
        elapsed = now_time - game_over_since
        alpha = min(1.0, elapsed / game_over_fade_ms)
        # 렌더 "GAME OVER" 텍스트에 알파 적용
        go_surf = mFont.render("GAME OVER", True, red)
        go_surf_alpha = pygame.Surface(go_surf.get_size(), pygame.SRCALPHA)
        go_surf_alpha.blit(go_surf, (0,0))
        # multiply alpha
        arr = pygame.surfarray.pixels_alpha(go_surf_alpha)
        arr[:] = (arr[:] * alpha).astype('uint8')
        del arr
        screen.blit(go_surf_alpha, (width // 2 - go_surf_alpha.get_width() // 2, height // 2 - 80))

        # 서브 텍스트 (검정 배경에 흰 글씨)
        go_sub = sFont.render("Press Enter to Restart  |  Esc -> Main Menu", True, white)
        final_score = sFont.render(f"Final Score: {score}", True, white)
        screen.blit(final_score, (width // 2 - final_score.get_width() // 2, height // 2 - 20))
        screen.blit(go_sub, (width // 2 - go_sub.get_width() // 2, height // 2 + 30))

        # 약간의 파티클/움직임: 화면 중앙에서 흩어지는 파티클
        if elapsed < 1500 and random.random() < 0.2:
            for _ in range(6):
                particles.append(Particle(width//2 + random.uniform(-60,60), height//2 + random.uniform(-20,20), (255,200,30)))

        for p in particles[:]:
            p.update(dt)
            if p.age >= p.life:
                particles.remove(p)
            else:
                p.draw()

        pygame.display.flip()
        continue

    if inputs.hardDrop:
        inputs.hardDrop = False

    # DAS/ARR 처리 (값 자체는 변경하지 않음, 고정값만 1.2배 적용됨)
    if inputs.moveLeft:
        if inputs.L_DAS_CNT >= inputs.DAS_VALUE and inputs.moveRight == False:
            if inputs.ARR_VALUE == 0:
                nowMino.eraseMino()
                while nowMino.isBlockedByMovement(-1,0):
                    nowMino.moveMino(-1,0)
                nowMino.drawMino()
                last_action = None
            elif inputs.L_ARR_CNT % inputs.ARR_VALUE == 0 and nowMino.isBlockedByMovement(-1,0):
                nowMino.eraseMino()
                nowMino.moveMino(-1,0)
                nowMino.drawMino()
                last_action = None
            inputs.L_ARR_CNT += 1
        inputs.L_DAS_CNT += 1
    if inputs.moveRight:
        if inputs.R_DAS_CNT >= inputs.DAS_VALUE and inputs.moveLeft == False:
            if inputs.ARR_VALUE == 0:
                nowMino.eraseMino()
                while nowMino.isBlockedByMovement(1,0):
                    nowMino.moveMino(1,0)
                nowMino.drawMino()
                last_action = None
            elif inputs.R_ARR_CNT % inputs.ARR_VALUE == 0 and nowMino.isBlockedByMovement(1,0):
                nowMino.eraseMino()
                nowMino.moveMino(1,0)
                nowMino.drawMino()
                last_action = None
            inputs.R_ARR_CNT += 1
        inputs.R_DAS_CNT += 1
    if inputs.softDrop:
        if inputs.SD_ARR_CNT % inputs.SD_ARR_VALUE == 0 and nowMino.isBlockedByMovement(0,1):
            nowMino.eraseMino()
            nowMino.moveMino(0,1)
            nowMino.drawMino()
            lock_timer = 0
            last_action = None
        inputs.SD_ARR_CNT += 1

    # --- 중력 타이머 처리 (자동 낙하)
    gravity_timer += dt_ms
    moved_by_gravity = False
    if gravity_timer >= gravity_interval_ms:
        gravity_timer -= gravity_interval_ms
        if nowMino.isBlockedByMovement(0,1):
            nowMino.eraseMino()
            nowMino.moveMino(0,1)
            nowMino.drawMino()
            moved_by_gravity = True
            lock_timer = 0
            is_touching = False
            last_action = None
        else:
            is_touching = True

    if not nowMino.isBlockedByMovement(0,1):
        is_touching = True
        lock_timer += dt_ms
    else:
        is_touching = False
        lock_timer = 0

    if is_touching and lock_timer >= lock_delay_ms:
        field.testMatrix = field.matrix.copy()
        last_locked_blocks = []
        for i in range(4):
            y = nowMino.data[i,1] + nowMino.data[5,1]
            x = nowMino.data[i,0] + nowMino.data[5,0]
            if 0 <= y < field.testMatrix.shape[0] and 0 <= x < field.testMatrix.shape[1]:
                field.testMatrix[y,x] = nowMino.data[4,0]
                last_locked_blocks.append((int(y), int(x)))
        last_locked_mino_type = int(nowMino.data[4,0])

        if last_locked_mino_type == 3:
            last_locked_pivot = (int(nowMino.data[2,1] + nowMino.data[5,1]), int(nowMino.data[2,0] + nowMino.data[5,0]))
        else:
            last_locked_pivot = None

        last_locked_was_tspin = False
        if last_locked_mino_type == 3 and last_action == 'rotate' and last_locked_pivot is not None:
            pr, pc = last_locked_pivot
            corners = [(pr-1,pc-1),(pr-1,pc+1),(pr+1,pc-1),(pr+1,pc+1)]
            occupied = 0
            for (cr,cc) in corners:
                if cr<0 or cr>=field.testMatrix.shape[0] or cc<0 or cc>=field.testMatrix.shape[1]:
                    occupied += 1
                else:
                    if field.testMatrix[cr,cc] != 0:
                        occupied += 1
            if occupied >= 3:
                last_locked_was_tspin = True

        # 고정
        field.matrix = field.testMatrix.copy()
        placed_pieces += 1

        # 라인 클리어 계산
        lines_cleared, tspin_detected = f.clearLines()

        # --- 상쇄 처리: 공격이 pending이면 지운만큼(또는 tspin 2배) 깎는다
        if attack_pending and lines_cleared > 0:
            cancel_amount = lines_cleared * (2 if tspin_detected else 1)
            attack_lines -= cancel_amount
            if attack_lines <= 0:
                attack_pending = False
                attack_ready = False
                attack_lines = 0
                # 상쇄되면 다음 공격 재스케줄
                next_attack_time = now_time + random.randint(15000, 30000)

        # 공격 적용 (빨간 준비 상태일 때만)
        if attack_pending and attack_ready and attack_lines > 0:
            apply_attack(attack_lines)
            attack_pending = False
            attack_ready = False
            next_attack_time = now_time + random.randint(15000, 30000)

        # 새 미노 스폰
        nowMino = spawn_new_mino_from_template(nowBag.nowQueue.pop(0))
        nowBag.generateBag()
        nowMino.drawMino()
        lock_timer = 0
        is_touching = False
        inputs.hold = False
        last_action = None
        last_locked_was_tspin = False

    if game_over:
        # 게임오버가 감지되는 첫 프레임에 타임스탬프를 세팅
        if game_over_since is None:
            game_over_since = now_time
        GameStart = False
        continue

    nowMino.drawGhost()

    # 파티클 업데이트
    for p in particles[:]:
        p.update(dt)
        if p.age >= p.life:
            particles.remove(p)

    # --- 그리기
    draw_starry_background(now_time)

    def draw_beveled_block(px, py, color):
        outer = pygame.Rect(px, py, 28, 28)
        border = 3
        inner = tuple(min(255, int(c + 60)) for c in color)
        edge = tuple(max(0, int(c * 0.6)) for c in color)
        pygame.draw.rect(screen, edge, outer)
        inner_rect = pygame.Rect(px + border, py + border, 28 - border*2, 28 - border*2)
        pygame.draw.rect(screen, inner, inner_rect)
        highlight_rect = pygame.Rect(px + 8, py + 8, 12, 12)
        pygame.draw.rect(screen, tuple(min(255, int(c + 120)) for c in color), highlight_rect)

    # 고스트: 테두리만, 두껍게
    for i in range(4):
        gx = nowMino.ghost[i,0] + nowMino.ghost[5,0]
        gy = nowMino.ghost[i,1] + nowMino.ghost[5,1]
        px = gx * CELL_SIZE + FIELD_X
        py = (gy - VISIBLE_TOP) * CELL_SIZE + FIELD_Y
        # 라인 굵기 5로 두껍게 테두리만 그림
        pygame.draw.rect(screen, lightgray, (px, py, 28, 28), 5)

    # 필드 블록
    for row in range(VISIBLE_TOP, 40):
        for col in range(10):
            val = field.matrix[row,col]
            if val != 0:
                px = col * CELL_SIZE + FIELD_X
                py = (row - VISIBLE_TOP) * CELL_SIZE + FIELD_Y
                draw_beveled_block(px, py, cell_Colors[val])

    # 배경격자
    for r in range(0,20):
        for c in range(10):
            gx = FIELD_X + c * CELL_SIZE
            gy = FIELD_Y + r * CELL_SIZE
            pygame.draw.rect(screen, (230,230,230), (gx, gy, CELL_SIZE, CELL_SIZE), 1)

    # next 및 hold
    for i in range(5):
        mino.nexts[i] = nowBag.nowQueue[i]
        for j in range(4):
            mv = mino.nexts[i][j]
            nx = 570 + 1 + mv[0] * 30
            ny = 60 + 1 + mv[1] * 30 + i * 90
            draw_beveled_block(nx, ny, cell_Colors[mino.nexts[i][4,0]])

    if hold[4,0] != -1:
        for j in range(4):
            hv = hold[j]
            hx = 60 + 1 + hv[0] * 30
            hy = 60 + 1 + hv[1] * 30
            draw_beveled_block(hx, hy, cell_Colors[hold[4,0]])

    # --- 공격 바: 세그먼트 높이를 필드 1칸(CELL_SIZE)으로 설정
    bar_x = FIELD_X - 36
    bar_y = FIELD_Y 
    bar_w = 28
    max_segments = 20
    bar_h = CELL_SIZE * max_segments 
    pygame.draw.rect(screen, (40,40,40), (bar_x, bar_y, bar_w, bar_h), 2)
    seg_h = bar_h / max_segments
    seg_w_inner = bar_w - 6
    for sidx in range(max_segments):
        seg_x = bar_x + 3
        seg_y = bar_y + bar_h - int((sidx+1) * seg_h) + 3
        rect = pygame.Rect(seg_x, seg_y, seg_w_inner, int(seg_h) - 4)
        pygame.draw.rect(screen, (20,20,20), rect, 1)  # segment outline

    if attack_pending:
        color = red if attack_ready else yellow
        for k in range(attack_lines):
            # 채우기: 아래에서 위로
            sidx = k
            seg_x = bar_x + 3
            seg_y = bar_y + bar_h - int((sidx+1) * seg_h) + 3
            fill_rect = pygame.Rect(seg_x + 1, seg_y + 1, seg_w_inner - 2, int(seg_h) - 6)
            pygame.draw.rect(screen, color, fill_rect)

    # HUD
    elapsed_sec = (pygame.time.get_ticks() - start_time) / 1000
    pps = round(placed_pieces / elapsed_sec, 2) if elapsed_sec > 0 else 0
    hud_lines = [
        f"Time: {int(elapsed_sec)}s",
        f"Lines Cleared: {total_clears}",
        f"PPS: {pps}",
        f"Score: {score}"
    ]
    for i, txt in enumerate(hud_lines):
        hud_text = sFont.render(txt, True, white)
        screen.blit(hud_text, (10, height - 120 + i * 24))

    # clear text (1초)
    if last_clear_text and pygame.time.get_ticks() - last_clear_time < 1000:
        top_entry, bottom_entry = last_clear_text
        # ALL CLEAR 특별 처리: 중앙에 두 줄로 크게 표시
        if top_entry and top_entry[0] == "ALL":
            # 중앙좌표 (필드 중앙)
            field_center_x = FIELD_X + (CELL_SIZE * 10) // 2
            field_center_y = FIELD_Y + (CELL_SIZE * 20) // 2
            # 위: ALL
            t_text, t_color = top_entry
            if t_color == "rainbow":
                draw_rainbow_text(t_text, mFont, field_center_x - mFont.size(t_text)[0]//2, field_center_y - 40)
            else:
                top_surf = mFont.render(t_text, True, t_color)
                screen.blit(top_surf, (field_center_x - top_surf.get_width() // 2, field_center_y - 40))
            # 아래: CLEAR
            b_text, b_color = bottom_entry
            if b_color == "rainbow":
                draw_rainbow_text(b_text, sFont, field_center_x - sFont.size(b_text)[0]//2, field_center_y + 5)
            else:
                bottom_surf = sFont.render(b_text, True, b_color)
                screen.blit(bottom_surf, (field_center_x - bottom_surf.get_width() // 2, field_center_y + 5))
        else:
            if top_entry:
                t_text, t_color = top_entry
                if t_color == "rainbow":
                    draw_rainbow_text(t_text, mFont, 60, 200)
                else:
                    top_surf = mFont.render(t_text, True, t_color)
                    screen.blit(top_surf, (60, 200))
            if bottom_entry:
                b_text, b_color = bottom_entry
                if b_color == "rainbow":
                    draw_rainbow_text(b_text, sFont, 60, 260)
                else:
                    bottom_surf = sFont.render(b_text, True, b_color)
                    screen.blit(bottom_surf, (60, 260))

    # 파티클
    for p in particles:
        p.draw()

    pygame.display.flip()

pygame.quit()
