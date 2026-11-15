# LunarLander PPO 학습 → 평가 → ONNX 변환 → FastAPI/Gradio 배포

한밭대학교 강화학습 기말 프로젝트. LunarLander-v3에서 PPO(CleanRL 기반)를 학습하고, 두 가지 행동 선택 방식과 관측 노이즈로 평가한 뒤, actor를 ONNX로 변환해 FastAPI 서버와 Gradio 화면으로 배포한다.

## 0. 이 README 읽는 법

1. 1~2장: 무엇을 했고 결과가 어떤지 (5분)
3~5장: 환경 설치와 폴더 구조
6장: 처음부터 똑같이 재현하는 명령 순서
7~8장: 결과 표, 로그가 어디에 있는지
9장: 한계와 주의 (읽어 두면 질의응답에 도움이 됨)

## 1. 한눈에 보는 결과

| 항목 | 결과 |
|---|---|
| 대표 seed (학번) | 20237141 |
| 선택한 설정 | try5 (대표 seed last_return 263.7, 통과 기준 200 초과) |
| 5 seed 평가, argmax (배포 방식) | 256.5 ± 4.0 |
| 5 seed 평가, 확률적 선택 | 230.4 ± 45.2 |
| 베이스라인 | 랜덤 -172.7, 무동작 -131.5 (성공률 둘 다 0%) |
| 노이즈 σ=0.1 / 0.2 | 228.5 / 110.9 (σ=0에서 256.5) |
| ONNX 변환 | 행동 일치율 1.0, logits 최대 오차 2.86e-06 |
| 배포 모델 | best seed 50237141 (policy.onnx, sha256 앞 8자리 2885ac92) |

## 2. 과제의 흐름 (8단계)

| 단계 | 하는 일 | 주요 파일 |
|---|---|---|
| 1 | 환경 설치와 점검 | requirements.txt |
| 2 | 랜덤·무동작 베이스라인 100회씩 | evaluate.py |
| 3 | PPO 학습. 대표 seed로 last_return > 200이 될 때까지 최대 5번 개선(try1~5), 이후 seed 4개 추가 | train_ppo.py |
| 4 | argmax와 확률적 선택으로 5개 모델 평가 | eval_ppo.py, policy.py |
| 5 | 관측 8개에 노이즈 σ를 더해 성능 하락 측정 | eval_noise.py |
| 6 | best 모델의 actor만 ONNX로 내보내기 | export_onnx.py |
| 7 | 변환 전후 출력 차이, 지연, 크기 비교 | check_onnx.py |
| 8 | FastAPI 서버, 클라이언트, Gradio 시연, 시연 사례 탐색 | serve.py, client.py, app.py, demo_check.py |

각 단계의 결과 파일이 다음 단계의 입력이므로 순서대로 진행한다.

## 3. 실험 환경

| 항목 | 값 |
|---|---|
| 운영체제 | Ubuntu 22.04.5 LTS (WSL2, Windows 호스트) (커널 6.18.58.1-microsoft-standard-WSL2) |
| CPU | AMD Ryzen 5 7600 6-Core Processor (논리 코어 12개) |
| 메모리 | 15.4 GB (WSL에 할당된 값) |
| 학습 장치 | CPU만 사용 (PyTorch CPU 빌드) |
| Python | 3.11.18 |
| PyTorch | 2.9.1+cpu |
| Gymnasium | 1.2.2 |
| onnx / onnxruntime | 1.23.2 / 1.32.0 |
| FastAPI / Gradio | 0.163.0 / 6.32.0 |
| matplotlib | 3.11.2 (보고서 그림용, 과제 설치 목록 외 추가 설치) |

Windows PC에서 WSL2의 Ubuntu 22.04로 진행했다. 과제는 Ubuntu 22.04 기준이므로 다른 환경에서는 수정이 필요할 수 있다. 전체 버전은 requirements.txt에 있다.

## 4. 설치 (1단계)

```bash
sudo apt update
sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.11-venv python3.11-dev g++
python3.11 -m venv .venv && source .venv/bin/activate
python -m pip install swig==4.3.1 setuptools==65.5.0 wheel==0.48.0
python -m pip install --no-build-isolation box2d-py==2.3.5
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install gymnasium==1.2.2 pydantic==2.11.10 onnxscript==0.5.6 onnx-ir==0.1.12 onnx onnxruntime fastapi uvicorn gradio requests imageio imageio-ffmpeg tensorboard pygame tyro
python -m pip install matplotlib
python -m pip freeze > requirements.txt
python -m pip check
```

환경 점검 (ENV OK가 나오면 통과):

```bash
python -c "import gymnasium as gym, torch, onnx, gradio, onnxruntime, onnxscript, fastapi, imageio_ffmpeg; e = gym.make('LunarLander-v3', render_mode='rgb_array'); o, _ = e.reset(seed=0); assert o.shape == (8,); assert e.action_space.n == 4 and e.render().ndim == 3; e.step(0); e.close(); print('ENV OK')"
```

## 5. 폴더 구조

```
project/
├─ README.md, requirements.txt
├─ common.py            공통 함수 (경로, CSV/JSON 저장, 학습 기록 병합)
├─ evaluate.py          평가 함수 run_episodes와 베이스라인
├─ ppo.py               CleanRL 원본 (commit e421c2e)
├─ train_ppo.py         ppo.py를 수정한 학습 코드 (최종 = try5 설정)
├─ policy.py            actor 로딩, argmax/확률적 정책
├─ eval_ppo.py          4단계 평가
├─ eval_noise.py        5단계 노이즈 평가
├─ export_onnx.py       6단계 ONNX 변환
├─ check_onnx.py        7단계 변환 검증, 지연, 크기
├─ serve.py             8단계 FastAPI 서버
├─ client.py            서버 호출 클라이언트
├─ app.py               Gradio 시연 화면
├─ demo_check.py        시연용 성공/미달 사례 탐색
├─ validate_results.py  제출 전 결과 파일 검증
├─ make_report.py       보고서용 그림과 표 생성
├─ make_readme.py       이 README 생성
└─ artifacts/
   ├─ smoke/            짧은 학습 점검 (제출 제외)
   └─ team/
      ├─ baseline/results/       베이스라인 CSV 2개
      ├─ try1 ~ try5/            try별 결과 (train_ppo.py 스냅샷, runs/, models/, results/)
      └─ selected_experiment.json   선택한 try (try5)
```

try5 폴더 안:

| 경로 | 내용 |
|---|---|
| train_ppo.py | try5 때 사용한 코드 스냅샷 |
| runs/ | TensorBoard 로그 (seed 5개) |
| models/ | 학습 체크포인트 .pt와 .json, best.pt, actor.pt, policy.onnx |
| results/ | training.json, eval_*.csv, evaluation.json, noise.json, noise/, export.json, conversion.json, observations.npy, demo_cases.csv |
| videos/ | 시연 영상 mp4 |
| report/ | 보고서용 그림 4개와 report_tables.md |

## 6. 재현 순서 (실제로 쓴 명령)

모든 명령은 저장소 폴더에서 가상환경을 켠 상태(`source .venv/bin/activate`)로 실행한다.

### 2단계 베이스라인

```bash
export EXP=artifacts/team/baseline
python evaluate.py --output "$EXP"
```

평가 seed는 10000~10099, 랜덤 행동 seed는 12345로 고정이다. 결과는 baseline_random.csv, baseline_noop.csv (각 100행).

### 3단계 ① 코드 준비와 짧은 점검

```bash
wget -O ppo.py https://raw.githubusercontent.com/vwxyzjn/cleanrl/e421c2e/cleanrl/ppo.py
cp ppo.py train_ppo.py
```

train_ppo.py에서 아래를 수정했다 (`diff ppo.py train_ppo.py`로 확인 가능).

- A: SyncVectorEnv에 `autoreset_mode=gym.vector.AutoresetMode.SAME_STEP` 추가
- B: 시간 제한으로만 끝난 env는 final_obs의 γ·V를 학습용 보상에 더함 (기록은 원래 보상)
- C: final_info의 "_episode"가 True인 env의 episode return을 TensorBoard와 ep_log에 기록
- D: 학습 후 체크포인트 저장, last_return(마지막 10만 스텝 평균 return) 계산, record_run으로 training.json 기록
- 출력 폴더: Args에 `output_dir` 추가 (`--output-dir`)
- 기본값을 RL Zoo 설정으로: env_id=LunarLander-v3, num_envs=16, num_steps=1024, num_minibatches=256, update_epochs=4, gamma=0.999, gae_lambda=0.98, ent_coef=0.01, total_timesteps=1000000

```bash
python train_ppo.py --seed 1 --total-timesteps 16384 --output-dir artifacts/smoke
```

### 3단계 ② 대표 seed 학습과 성능 개선 (try1~5)

대표 seed는 20237141이다. try마다 train_ppo.py의 Args 기본값만 바꾸고, 새 폴더에 스냅샷을 복사한 뒤 학습했다.

```bash
EXP=artifacts/team/try1      # try2, try3, try4, try5 순서로 반복
mkdir -p "$EXP" && cp train_ppo.py "$EXP/"
python train_ppo.py --seed 20237141 --output-dir "$EXP"
```

try별로 바꾼 설정 (바뀐 줄 외에는 모두 같음):

| try | update_epochs | anneal_lr | clip_vloss | vf_coef | 비고 |
|---|---|---|---|---|---|
| try1 | 4 | True | True | 0.5 | Zoo 설정 그대로 |
| try2 | 10 | True | True | 0.5 | update_epochs만 변경 |
| try3 | 10 | False | True | 0.5 | try2에서 anneal_lr만 변경 |
| try4 | 10 | True | False | 0.5 | try2에서 clip_vloss만 변경 (try3의 변경은 되돌림) |
| try5 | 10 | True | False | 1.0 | try4에서 vf_coef만 변경 |

각 try 폴더의 train_ppo.py가 그 시점의 코드 스냅샷이다. 특정 try를 다시 돌리려면 그 폴더의 스냅샷을 루트로 복사해서 실행한다 (루트의 train_ppo.py가 덮어써지므로 끝나면 try5 것으로 되돌릴 것).

```bash
cp artifacts/team/tryN/train_ppo.py .
python train_ppo.py --seed 20237141 --output-dir artifacts/team/tryN
```

판정 값(대표 seed의 last_return) 확인과 선택 기록:

```bash
python - artifacts/team/try5/results/training.json << 'EOF'
import json, sys
for r in json.load(open(sys.argv[1]))["runs"]:
    print(r["seed"], r["last_return"])
EOF
echo '{"directory": "try5"}' > artifacts/team/selected_experiment.json
```

### 3단계 ③ seed 4개 추가 학습

선택한 try5의 코드와 설정 그대로, seed = 20237141 + 10,000,000 × k (k=1~4)로 학습한다 (30237141, 40237141, 50237141, 60237141).

```bash
BASE_SEED=20237141
EXP=artifacts/team/try5
cp "$EXP/train_ppo.py" .
for k in 1 2 3 4; do
  python train_ppo.py --seed $((BASE_SEED + k * 10000000)) --output-dir "$EXP"
done
```

### 4단계 평가

```bash
export EXP=artifacts/team/try5
python eval_ppo.py --output "$EXP"
```

모델 5개 × 2방식(argmax, 확률적) × 100회. 결과: eval_<seed>_<방식>.csv 10개, evaluation.json, models/best.pt (argmax 평균이 가장 높은 모델).

### 5단계 노이즈 평가

```bash
python eval_noise.py --output "$EXP"
```

σ = 0, 0.01, 0.05, 0.1, 0.2 × 모델 5개 × 100회. 결과: noise.json, noise/ 폴더 csv 25개.

### 6~7단계 ONNX 변환과 검증

```bash
python export_onnx.py --output "$EXP"
python check_onnx.py --output "$EXP"
```

### 8단계 서버와 시연 (터미널 2개)

터미널 A (서버, 켜 둔 채로 유지):

```bash
export EXP=artifacts/team/try5
LUNAR_MODEL="$EXP/models/policy.onnx" python -m uvicorn serve:app --host 127.0.0.1 --port 8000
```

터미널 B:

```bash
export EXP=artifacts/team/try5
curl -s http://127.0.0.1:8000/model
cat "$EXP/results/export.json"            # 두 sha256이 같아야 함
curl -s -X POST http://127.0.0.1:8000/act -H "Content-Type: application/json" -d '{"obs": [0, 0, 0, 0, 0, 0, 1, 1]}'
curl -s -o /dev/null -w "%{http_code}\n" -X POST http://127.0.0.1:8000/act -H "Content-Type: application/json" -d '{"obs": [0, 0, 0]}'   # 422
python client.py --url http://127.0.0.1:8000
python app.py --url http://127.0.0.1:8000 --port 7860 --output "$EXP/videos"   # 브라우저에서 127.0.0.1:7860, 확인 후 Ctrl+C
python demo_check.py --url http://127.0.0.1:8000 --output "$EXP"
```

### 제출 전 검증과 보고서 재료

```bash
python validate_results.py --output "$EXP"
python make_report.py
```

## 7. 결과

보고서용 표 전체는 `artifacts/team/try5/report/report_tables.md`, 그림은 같은 폴더에 있다.

### 대표 seed try별 기록 (3단계 ②)

| try | 변경 | last_return | 판정 |
|---|---|---|---|
| try1 | Zoo 설정 그대로 | 88.0 | 미달 |
| try2 | update_epochs 4 → 10 | 113.7 | 미달 |
| try3 | anneal_lr False | 96.5 | 미달 |
| try4 | clip_vloss False (try2 기준) | 186.2 | 미달 |
| try5 | vf_coef 1.0 (try4 기준) | 263.7 | 통과 |

변경 이유는 당시의 가설이다. 곡선이 50만 스텝 이후 100 근처에서 정체해 학습이 막혔다고 보았고, try4에서는 보상 규모가 큰 환경에서 가치함수 clipping(±0.2)이 critic 학습을 막을 수 있다고 가정했다. 효과가 컸지만 clip_vloss만 따로 반복 검증한 것은 아니다.

### 4단계: 두 행동 선택 방식 (5개 모델 × 100회, 평가 seed 10000~10099)

| 학습 seed | argmax 평균 | 표준편차 | 확률적 평균 | 표준편차 |
|---|---|---|---|---|
| 20237141 | 254.4 | 21.4 | 263.7 | 44.5 |
| 30237141 | 260.9 | 20.9 | 260.9 | 44.1 |
| 40237141 | 252.3 | 25.4 | 146.3 | 40.3 |
| 50237141 | 261.8 | 21.0 | 219.7 | 57.4 |
| 60237141 | 253.4 | 21.8 | 261.7 | 42.4 |
| 5개 평균 ± 표준편차 | 256.5 ± 4.0 | 22.1 | 230.4 ± 45.2 | 45.7 |

- 200 미만 에피소드: argmax 2/500 (terminated 0, truncated 2), 확률적 155/500 (terminated 20, truncated 135).
- 확률적 선택의 편차가 큰 것은 seed 40237141 한 모델(146.3)의 영향이 크다. 200 미만의 대부분이 추락이 아니라 시간 초과(truncated)였다.
- 표준편차 열은 모델별 100회 에피소드 간 값이고, 평균 열의 ±는 5개 모델 평균 사이 값이다 (n으로 나눔).
- best 모델은 argmax 평균이 가장 높은 50237141이다. 고를 때 쓴 100회 값이라 실제보다 약간 높을 수 있다.

### 5단계: 관측 노이즈 (argmax, 5개 모델 × 100회)

| σ | 평균 return | 표준편차(5개 모델 간) | σ=0 대비 |
|---|---|---|---|
| 0.0 | 256.5 | 4.0 | +0.0 |
| 0.01 | 256.0 | 4.8 | -0.6 |
| 0.05 | 259.4 | 8.6 | +2.8 |
| 0.1 | 228.5 | 45.4 | -28.1 |
| 0.2 | 110.9 | 23.4 | -145.7 |

σ=0의 값은 4단계 argmax 평균(256.5)과 같다. σ가 0.05까지는 차이가 표준편차 안쪽이고, 0.1부터 모델 간 편차가 커지며 하락하고, 0.2에서는 절반 이하가 된다. 노이즈는 다리 접촉(0/1) 값에도 같은 크기로 더해졌다.

### 6~7단계: 변환 검증 (best 모델, 같은 PC, CPU 스레드 1개)

| 항목 | PyTorch | ONNX |
|---|---|---|
| actor 크기 (KiB) | 22.1 | 25.9 |
| 지연 평균·p95 (ms) | 0.0232 · 0.0326 | 0.0058 · 0.0072 |
| 행동 일치율 (관측 10,000개) | 기준 | 1.0 |
| 최대 오차 (logits) | 기준 | 2.86e-06 |
| 평균 오차 (logits) | 기준 | 1.14e-07 |

- ONNX가 지연은 약 4배 빠르고 파일은 더 크다. 파일이 큰 이유는 가중치 외에 계산 그래프와 메타데이터가 함께 저장되기 때문으로 추정한다.
- 서버의 /model sha256과 results/export.json의 policy_sha256이 일치함을 확인했다.

### 8단계: 시연과 사례 탐색

- Gradio에서 seed 10000을 두 번 실행해 영상 파일이 따로 생기고, return 252.7이 PyTorch 평가(eval_50237141_argmax.csv, 252.687)와 일치함을 확인했다.
- demo_check.py로 평가 seed 10000~10099와 20000~20199, 총 300개를 확인했으나 return 200 미만 사례는 없었다. 성공 사례(seed 10000, return 252.7)만 영상으로 저장했고, 탐색 기록은 results/demo_cases.csv에 있다. 이 탐색 결과는 평가 평균에 섞지 않았다.

## 8. 로그 매핑 (try 폴더 → seed → runs/ → 모델 파일)

경로는 artifacts/team/ 기준이다. 학습 시간은 로그 폴더 이름의 시작 시각과 기록 파일 저장 시각의 차이다. 확인하려면 `tensorboard --logdir artifacts/team`을 실행하고 브라우저에서 localhost:6006을 연다.

| try | 학습 seed | runs/ 폴더 | 모델 파일 | last_return | 학습 시간(분) |
|---|---|---|---|---|---|
| try1 | 20237141 | try1/runs/LunarLander-v3__train_ppo__20237141__1794725280/ | try1/models/LunarLander-v3__train_ppo__20237141__1794725280.pt | 88.0 | 2.7 |
| try2 | 20237141 | try2/runs/LunarLander-v3__train_ppo__20237141__1794727424/ | try2/models/LunarLander-v3__train_ppo__20237141__1794727424.pt | 113.7 | 5.7 |
| try3 | 20237141 | try3/runs/LunarLander-v3__train_ppo__20237141__1794728176/ | try3/models/LunarLander-v3__train_ppo__20237141__1794728176.pt | 96.5 | 5.7 |
| try4 | 20237141 | try4/runs/LunarLander-v3__train_ppo__20237141__1794728686/ | try4/models/LunarLander-v3__train_ppo__20237141__1794728686.pt | 186.2 | 9.8 |
| try5 | 20237141 | try5/runs/LunarLander-v3__train_ppo__20237141__1794731637/ | try5/models/LunarLander-v3__train_ppo__20237141__1794731637.pt | 263.7 | 15.1 |
| try5 | 30237141 | try5/runs/LunarLander-v3__train_ppo__30237141__1794733206/ | try5/models/LunarLander-v3__train_ppo__30237141__1794733206.pt | 263.3 | 11.1 |
| try5 | 40237141 | try5/runs/LunarLander-v3__train_ppo__40237141__1794733874/ | try5/models/LunarLander-v3__train_ppo__40237141__1794733874.pt | 140.6 | 8.7 |
| try5 | 50237141 | try5/runs/LunarLander-v3__train_ppo__50237141__1794734398/ | try5/models/LunarLander-v3__train_ppo__50237141__1794734398.pt | 186.9 | 5.3 |
| try5 | 60237141 | try5/runs/LunarLander-v3__train_ppo__60237141__1794734715/ | try5/models/LunarLander-v3__train_ppo__60237141__1794734715.pt | 252.0 | 5.1 |

## 9. 한계와 주의

- 각 try는 대표 seed 하나의 결과라서 try 사이의 차이가 설정 효과인지 우연인지 구분할 수 없다. 특히 try2, try3 사이 같은 작은 차이는 우연 범위일 수 있다.
- RL Zoo 원본과 완전히 같은 실험은 아니다. Zoo에 없는 값은 CleanRL 기본값(학습률 2.5e-4 선형 감소 등)을 사용했다.
- 평가 seed(10000~10099)는 best 모델을 고를 때도 쓰였으므로 best 모델의 성능은 약간 높게 나올 수 있다.
- 시연 사례 탐색에서 미달 사례를 찾지 못해 성공 사례만 확보했다.
- 지연과 크기 수치는 이 PC에서 CPU 스레드 1개로 잰 값이며 다른 PC에서는 달라진다.
- 짧은 학습 점검(artifacts/smoke)은 본 실험 결과가 아니며 제출 대상이 아니다.

## 10. 용어

- return: 한 에피소드의 보상 합. 성공은 return 200 이상.
- last_return: 학습 마지막 10만 스텝 동안 끝난 에피소드의 평균 return. 대표 seed 통과 판정 값.
- try: 설정을 바꿔 대표 seed를 다시 학습한 단위.
- argmax / 확률적 선택: logits가 가장 큰 행동을 고름 (배포 방식) / softmax 확률로 행동을 뽑음 (학습 방식).
- σ: 관측에 더하는 정규분포 잡음의 표준편차.
- actor / critic: 행동을 고르는 신경망 / 상태 가치를 추정하는 신경망. 배포에는 actor만 쓴다.
- logits: 행동 4개 각각의 출력값.

## 11. 팀과 AI 사용

- 팀원과 역할: TODO (이름과 담당 내용을 적을 것)
- AI 대화 증빙: TODO (사용한 대화의 공유 링크 또는 전체 내보내기 파일 위치)
