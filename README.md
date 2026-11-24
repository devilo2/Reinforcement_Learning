# LunarLander PPO 학습 → 평가 → ONNX 변환 → FastAPI/Gradio 배포

한밭대학교 강화학습 기말 프로젝트. `LunarLander-v3` 환경에서 PPO(CleanRL 기반)를 학습하고, 두 가지 행동 선택 방식(`argmax` vs 확률적 선택)과 관측 노이즈($\sigma$)로 정책을 평가한 뒤, 최적 모델의 Actor 신경망만 ONNX로 변환해 FastAPI 추론 서버와 Gradio 웹 화면으로 배포한다.

---

## 0. 이 README 읽는 법 (팀원 및 채점자 가이드)

1. **1~2장**: 프로젝트 전체 요약과 8단계 파이프라인 흐름 (5분 요약)
2. **3~5장**: 실험 PC 사양, 패키지 설치 명령, 저장소 폴더/파일 구조
3. **6장**: 1단계부터 8단계까지 처음부터 똑같이 재현하는 터미널 명령어 순서
4. **7~8장**: 단계별 실험 결과 표, 수치 해석(왜 이런 결과가 나왔는지), TensorBoard 로그 매핑 표
5. **9~11장**: 실험의 한계와 주의사항(발표 질의응답 핵심), 용어 정리, 팀원 역할 분담 및 AI 사용 증빙

> **⚠️ 팀원 필독 주의사항**
> - 모든 학습과 평가(1~8단계)는 이미 완료되어 `artifacts/team/try5/`에 저장되어 있으며 `validate_results.py` 검증을 통과했다.
> - **절대 학습(`train_ppo.py`)이나 평가 스크립트를 다시 덮어써서 돌리지 말 것!** (다시 돌리면 타임스탬프, 파일 해시, 지연 시간 등이 바뀌어 보고서 수치 및 TensorBoard 로그와 불일치하게 되어 감점됨)
> - 보고서와 발표 자료의 모든 숫자는 반드시 이 `README.md` 7~8장 및 `artifacts/team/try5/report/report_tables.md`의 값을 그대로 사용한다.
> - 7장의 "이유 분석" 중 `(가설)`, `(추정)`이 붙은 문장은 실험으로 확인한 사실이 아니다. 보고서와 발표에서도 가설로 표현한다.

---

## 1. 한눈에 보는 핵심 결과

| 항목 | 결과 요약 |
|---|---|
| **대표 seed (학번)** | `20237141` (추가 4개 seed: `30237141`, `40237141`, `50237141`, `60237141`) |
| **선택한 실험 설정** | **`try5`** (대표 seed `last_return = 263.7`, 과제 통과 기준 200 초과 달성) |
| **베이스라인 (100회)** | 랜덤 정책 `-172.7 ± 105.0` (성공률 0%), 무동작(noop) 정책 `-131.5 ± 44.0` (성공률 0%) |
| **4단계: 5-seed 평가 (`argmax`, 배포 방식)** | **`256.5 ± 4.0`** (에피소드 간 표준편차 평균 `22.1`, 500회 중 200 미만 단 2회) |
| **4단계: 5-seed 평가 (확률적 선택, 학습 방식)** | **`230.4 ± 45.2`** (에피소드 간 표준편차 평균 `45.7`, 500회 중 200 미만 155회) |
| **5단계: 관측 노이즈 ($\sigma = 0.0 \rightarrow 0.05$)** | `256.5 ± 4.0` ($\sigma=0.0$) $\rightarrow$ `256.0 ± 4.8` ($\sigma=0.01$) $\rightarrow$ `259.4 ± 8.6` ($\sigma=0.05$, 오차 범위 내 유지) |
| **5단계: 관측 노이즈 ($\sigma = 0.1 \rightarrow 0.2$)** | `228.5 ± 45.4` ($\sigma=0.1$, `40237141` 모델 급락으로 편차 증가) $\rightarrow$ `110.9 ± 23.4` ($\sigma=0.2$, 성능 붕괴) |
| **6~7단계: ONNX 변환 및 검증** | 행동 일치율 **`1.0`** (10,000개 관측), logits 최대 오차 `2.86e-06`, 평균 오차 `1.14e-07` |
| **7단계: 추론 지연 (CPU 1스레드) 및 크기** | 지연 평균·p95: PyTorch `0.0232 · 0.0326 ms` vs ONNX **`0.0058 · 0.0072 ms` (약 4배 빠름)** / 크기: `22.1 KiB` vs `25.9 KiB` |
| **8단계: 최종 배포 모델** | Best seed **`50237141`** (`argmax` 평균 `261.8`), `policy.onnx` 해시 앞 8자리 `2885ac92` |

---

## 2. 과제의 흐름 (8단계 파이프라인)

각 단계의 결과 파일이 다음 단계의 입력이 되므로 반드시 순서대로 진행하도록 설계되어 있다.

| 단계 | 목표 및 수행 내용 | 주요 스크립트 | 핵심 출력 파일 (`$EXP` = `artifacts/team/try5`) |
|---|---|---|---|
| **1단계** | Ubuntu 22.04 환경 구축, 패키지 버전 고정 및 `ENV OK` 점검 | - | `requirements.txt` |
| **2단계** | 랜덤(seed 12345 고정)·무동작(행동 0) 정책 각 100회(seed 10000~10099) 평가 | `evaluate.py` | `baseline/results/baseline_random.csv`, `baseline_noop.csv` |
| **3단계** | CleanRL `ppo.py`를 Gymnasium 1.2.2에 맞게 수정(`train_ppo.py`). 대표 seed로 `last_return > 200`까지 최대 5회 개선(`try1~5`) 후 선택한 설정(`try5`)으로 seed 4개 추가 학습 | `train_ppo.py` | `selected_experiment.json`, `results/training.json`, `models/*.pt`, `runs/` |
| **4단계** | 5개 모델을 `argmax`(결정적)와 `sample`(확률적) 두 방식으로 각 100회 평가 및 비교, 최고 모델 `best.pt` 선정 | `eval_ppo.py`, `policy.py` | `results/eval_<seed>_<mode>.csv`(10개), `results/evaluation.json`, `models/best.pt` |
| **5단계** | 관측 8개 모두에 가우시안 노이즈 $\sigma \in \{0, 0.01, 0.05, 0.1, 0.2\}$ 주입 후 5개 모델 × 100회 평가 (`argmax`) | `eval_noise.py` | `results/noise/<seed>_sigma<σ>.csv`(25개), `results/noise.json` |
| **6단계** | `best.pt`에서 추론에 쓰이는 Actor 신경망만 추출해 단일 파일 `policy.onnx`(opset 18)로 변환 및 해시 기록 | `export_onnx.py` | `models/policy.onnx`, `results/export.json` |
| **7단계** | 실제 관측 10,000개(seed 0 수집)에서 PyTorch vs ONNX 출력 오차·행동 일치율 검증, CPU 1스레드 추론 지연(1,000회) 및 크기 비교 | `check_onnx.py` | `models/actor.pt`, `results/observations.npy`, `results/conversion.json` |
| **8단계** | FastAPI 추론 서버(`/model`, `/act`, 422 입력 검증), 클라이언트 호출, Gradio 웹 시연 및 성공/미달 사례 300개 seed 탐색 | `serve.py`, `client.py`, `app.py`, `demo_check.py` | `results/demo_cases.csv`, `videos/*.mp4` |

---

## 3. 실험 환경

| 항목 | 값 |
|---|---|
| **운영체제** | Ubuntu 22.04.5 LTS (WSL2, Windows 호스트 / 커널 `6.18.58.1-microsoft-standard-WSL2`) |
| **CPU** | AMD Ryzen 5 7600 6-Core Processor (논리 코어 12개) |
| **메모리** | 15.4 GB (WSL2 할당 메모리) |
| **학습 및 추론 장치** | CPU 전용 (PyTorch CPU 빌드 사용) |
| **Python / PyTorch** | `3.11.18` / `2.9.1+cpu` |
| **Gymnasium / Box2D** | `1.2.2` / `box2d-py 2.3.5` (`swig 4.3.1`) |
| **ONNX 계열** | `onnx 1.23.2`, `onnxruntime 1.32.0`, `onnxscript 0.5.6`, `onnx-ir 0.1.12` |
| **서버 및 웹 데모** | `pydantic 2.11.10`, `fastapi 0.163.0`, `gradio 6.32.0` |
| **시각화 (추가 설치)** | `matplotlib 3.11.2` (`make_report.py` 보고서 그래프 생성용) |

Windows PC의 WSL2 Ubuntu 22.04 환경에서 모든 학습과 평가를 단일 PC 기준으로 수행했다. 전체 패키지 고정 버전은 `requirements.txt`에 기록되어 있다.

---

## 4. 설치 및 환경 점검 (1단계)

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

환경 점검 명령어 (`ENV OK`가 출력되면 정상 통과):

```bash
python -c "import gymnasium as gym, torch, onnx, gradio, onnxruntime, onnxscript, fastapi, imageio_ffmpeg; e = gym.make('LunarLander-v3', render_mode='rgb_array'); o, _ = e.reset(seed=0); assert o.shape == (8,); assert e.action_space.n == 4 and e.render().ndim == 3; e.step(0); e.close(); print('ENV OK')"
```

---

## 5. 폴더 및 파일 구조

```text
project/
├─ README.md                 # 프로젝트 전체 문서 (재현 순서, 결과 표, 로그 매핑, 분석)
├─ requirements.txt          # pip freeze 패키지 버전 기록
├─ common.py                 # 공통 함수 (exp_dirs, CSV/JSON 입출력, record_run, merge_runs, check_same_config)
├─ evaluate.py               # 공통 평가 제너레이터 run_episodes() 및 2단계 베이스라인 평가
├─ ppo.py                    # CleanRL PPO 원본 코드 (commit e421c2e)
├─ train_ppo.py              # Gymnasium 1.2.2 호환 수정(A~D) 및 RL Zoo/try5 설정이 반영된 PPO 학습 코드
├─ policy.py                 # Actor 가중치 분리 로더(load_actor) 및 PyTorch 정책 래퍼(torch_policy)
├─ eval_ppo.py               # 4단계: 5개 모델 × (argmax, sample) 100회 평가 및 best.pt 선정
├─ eval_noise.py             # 5단계: 관측 노이즈(σ=0~0.2) 5개 모델 × 100회 평가
├─ export_onnx.py            # 6단계: best.pt의 Actor를 policy.onnx(opset 18)로 변환 및 해시 저장
├─ check_onnx.py             # 7단계: 관측 10,000개 출력 오차·행동 일치율 검증, 1스레드 지연·크기 측정
├─ serve.py                  # 8단계: FastAPI 추론 서버 (/model 해시 조회, /act 추론 및 422 입력 검증)
├─ client.py                 # 8단계: FastAPI 호출 클라이언트 (Session 재사용, timeout=5, 상태 코드 검증)
├─ app.py                    # 8단계: Gradio 시연 웹 인터페이스 (서버 연동 에피소드 실행 및 MP4 녹화)
├─ demo_check.py             # 8단계: 발표용 성공(>=200)/미달(<200) 사례 시드 탐색 스크립트
├─ validate_results.py       # 제출 전 모든 결과 파일 존재·행 개수·설정 동일성·해시 일치 자동 검증
├─ make_report.py            # 보고서용 그래프 4종(PNG) 및 마크다운 요약표(report_tables.md) 생성
├─ make_readme.py            # README 초안 생성 스크립트 (현재 README는 직접 수정한 최종본이므로 다시 실행하면 덮어써지니 실행하지 말 것)
└─ artifacts/
   ├─ smoke/                 # 3단계 짧은 학습 점검 폴더 (제출용 압축 시 제외)
   └─ team/
      ├─ baseline/results/   # baseline_random.csv, baseline_noop.csv (각 100행)
      ├─ try1 ~ try4/        # 대표 seed(20237141) 개선 과정 (train_ppo.py 스냅샷, runs/, models/, results/)
      ├─ try5/               # 최종 선택된 실험 폴더 (5개 seed 전체 학습 및 4~8단계 모든 결과 포함)
      └─ selected_experiment.json  # {"directory": "try5"}
```

최종 선택된 `artifacts/team/try5/` 내부 구성:

| 하위 경로 | 포함 파일 및 설명 |
|---|---|
| `train_ppo.py` | `try5` 학습 시점의 코드 스냅샷 |
| `runs/` | 5개 학습 seed(`20237141` ~ `60237141`) 각각의 TensorBoard 이벤트 로그 폴더 |
| `models/` | 5개 모델 체크포인트(`.pt`)와 메타데이터(`.json`), `best.pt`, `actor.pt`, `policy.onnx` |
| `results/` | `training.json`, `eval_*.csv`(10개), `evaluation.json`, `noise.json`, `noise/*.csv`(25개), `export.json`, `conversion.json`, `observations.npy`, `demo_cases.csv` |
| `videos/` | Gradio 시연 화면으로 생성한 착륙 성공 MP4 영상 3개 (`seed10000_*.mp4`) |
| `report/` | 보고서용 그래프 4개(`tries_curve.png`, `learning_curve_5seeds.png`, `action_modes.png`, `noise.png`) 및 `report_tables.md` |

---

## 6. 단계별 재현 순서 (실제 실행 명령어)

모든 명령은 프로젝트 루트(`~/project`)에서 가상환경을 활성화(`source .venv/bin/activate`)한 상태로 실행한다.

### 2단계: 베이스라인 측정

```bash
export EXP=artifacts/team/baseline
python evaluate.py --output "$EXP"
```
- 평가 seed는 `10000~10099`(100회, 이후 모든 평가에 공통), 랜덤 정책의 행동 난수 seed는 `12345` 고정이다.

### 3단계 ①: CleanRL 코드 수정 및 짧은 학습 점검 (Smoke Test)

```bash
wget -O ppo.py https://raw.githubusercontent.com/vwxyzjn/cleanrl/e421c2e/cleanrl/ppo.py
cp ppo.py train_ppo.py
```

CleanRL 원본 `ppo.py`(Gymnasium 0.29.1 기준)를 최신 `gymnasium==1.2.2`에서 동작하고 결과를 기록하도록 아래 **A~D** 및 옵션을 수정했다 (`diff ppo.py train_ppo.py`로 확인 가능):
- **A (환경 자동 리셋 모드)**: `SyncVectorEnv`에 `autoreset_mode=gym.vector.AutoresetMode.SAME_STEP`을 추가하여 에피소드 종료 즉시 같은 스텝에서 `final_obs`와 `final_info`가 생성되도록 함.
- **B (시간 초과 부트스트래핑)**: 추락/착륙(`terminations`)이 아니라 1,000스텝 시간 제한(`truncations & ~terminations`)으로만 끝난 환경은 `final_obs`에서 추정한 가치 $\gamma V(s_{\text{final}})$를 학습용 보상에 더해줌 (단, 로그 기록은 원래 에피소드 보상 합 그대로 기록).
- **C (에피소드 종료 기록)**: `infos["final_info"]`에서 `"_episode"`가 `True`인 환경 인덱스를 찾아 `fi["episode"]["r"][i]`를 TensorBoard(`charts/episodic_return`)와 `ep_log` 리스트에 기록.
- **D (체크포인트 및 판정값 저장)**: 학습 종료 후 `torch.save(agent.state_dict(), ckpt)`로 모델을 저장하고, 마지막 10만 스텝(`global_step >= 900,000`) 동안 끝난 에피소드들의 평균인 `last_return`을 계산해 `record_run()`으로 `models/<run_name>.json` 및 `results/training.json`에 기록.
- **RL Zoo 기본값 적용 (`try1` 기준)**: `env_id="LunarLander-v3"`, `num_envs=16`, `num_steps=1024`, `num_minibatches=256`(미니배치 크기 $16 \times 1024 \div 256 = 64$), `update_epochs=4`, `gamma=0.999`, `gae_lambda=0.98`, `ent_coef=0.01`, `total_timesteps=1000000`. 신경망 구조(`64-64 Tanh`)는 변경하지 않음.

```bash
# 짧은 학습 점검 (16,384 스텝, 제출 및 try 횟수에서 제외)
python train_ppo.py --seed 1 --total-timesteps 16384 --output-dir artifacts/smoke
```

### 3단계 ②: 대표 seed(`20237141`) 학습과 성능 개선 (`try1` ~ `try5`)

대표 학번 seed `20237141` 하나로 `last_return > 200`을 넘을 때까지 한 번에 설정을 하나씩만 바꾸며 5회 실험했다. 각 `try`마다 당시의 `train_ppo.py`를 해당 폴더에 스냅샷으로 복사해 두었다.

```bash
EXP=artifacts/team/try1      # try1부터 try5까지 설정 변경 후 각각 실행
mkdir -p "$EXP" && cp train_ppo.py "$EXP/"
python train_ppo.py --seed 20237141 --output-dir "$EXP"
```

| try | `update_epochs` | `anneal_lr` | `clip_vloss` | `vf_coef` | `last_return` | 판정 및 변경 내용 요약 |
|---|---|---|---|---|---|---|
| **try1** | 4 | True | True | 0.5 | 88.0 | 미달 (RL Zoo 기본 설정 그대로) |
| **try2** | **10** | True | True | 0.5 | 113.7 | 미달 (`try1`에서 `update_epochs`만 4 $\rightarrow$ 10으로 변경) |
| **try3** | 10 | **False** | True | 0.5 | 96.5 | 미달 (`try2`에서 `anneal_lr`만 False로 변경 $\rightarrow$ 성능 하락으로 `try4`에서 복구) |
| **try4** | 10 | True | **False** | 0.5 | 186.2 | 미달 (`try2` 기준에서 `clip_vloss`만 False로 변경 $\rightarrow$ 큰 폭 개선) |
| **try5** | 10 | True | False | **1.0** | **263.7** | **통과** (`try4` 기준에서 `vf_coef`만 0.5 $\rightarrow$ 1.0으로 변경, 최종 선택) |

특정 `tryN`을 재현하려면 해당 폴더에 보관된 스냅샷을 루트로 복사해 실행하면 된다 (실행 후 루트의 `train_ppo.py`는 다시 `try5` 것으로 되돌려 둘 것):

```bash
cp artifacts/team/tryN/train_ppo.py .
python train_ppo.py --seed 20237141 --output-dir artifacts/team/tryN
```

판정 값 확인 및 `try5` 최종 선택 기록:

```bash
python - artifacts/team/try5/results/training.json << 'EOF'
import json, sys
for r in json.load(open(sys.argv[1]))["runs"]:
    print(r["seed"], r["last_return"])
EOF
echo '{"directory": "try5"}' > artifacts/team/selected_experiment.json
```

### 3단계 ③: 선택한 `try5` 설정으로 4개 seed 추가 학습 (총 5개 모델)

`try5`의 코드와 설정 그대로 `seed = 20237141 + 10,000,000 × k` ($k=1, 2, 3, 4$) 총 4개의 추가 seed(`30237141`, `40237141`, `50237141`, `60237141`)를 학습했다.

```bash
BASE_SEED=20237141
EXP=artifacts/team/try5
cp "$EXP/train_ppo.py" .
for k in 1 2 3 4; do
  python train_ppo.py --seed $((BASE_SEED + k * 10000000)) --output-dir "$EXP"
done
```

### 4단계: 두 가지 행동 선택 방식 정책 평가 (`argmax` vs `sample`)

```bash
export EXP=artifacts/team/try5
python eval_ppo.py --output "$EXP"
```

### 5단계: 관측 노이즈($\sigma$) 환경 평가

```bash
python eval_noise.py --output "$EXP"
```

### 6~7단계: ONNX 내보내기 및 변환 검증 (출력 오차·지연·크기)

```bash
python export_onnx.py --output "$EXP"
python check_onnx.py --output "$EXP"
```

### 8단계: FastAPI 추론 서버와 Gradio 시연 및 사례 탐색 (터미널 2개 사용)

**터미널 A (FastAPI 추론 서버 실행 및 유지):**
```bash
source .venv/bin/activate
export EXP=artifacts/team/try5
LUNAR_MODEL="$EXP/models/policy.onnx" python -m uvicorn serve:app --host 127.0.0.1 --port 8000
```

**터미널 B (클라이언트 검증, Gradio 웹 시연, 성공/미달 사례 탐색):**
```bash
source .venv/bin/activate
export EXP=artifacts/team/try5

# 1) 서버 모델 해시(/model)와 export.json 해시 일치 확인
curl -s http://127.0.0.1:8000/model
cat "$EXP/results/export.json"

# 2) 정상 추론(HTTP 200) 및 잘못된 입력 형식 거절(HTTP 422) 확인
curl -s -X POST http://127.0.0.1:8000/act -H "Content-Type: application/json" -d '{"obs": [0, 0, 0, 0, 0, 0, 1, 1]}'
curl -s -o /dev/null -w "%{http_code}\n" -X POST http://127.0.0.1:8000/act -H "Content-Type: application/json" -d '{"obs": [0, 0, 0]}'

# 3) 클라이언트 스크립트 및 Gradio 웹 화면 실행 (브라우저 127.0.0.1:7860 접속해 2회 이상 실행 확인 후 Ctrl+C)
python client.py --url http://127.0.0.1:8000
python app.py --url http://127.0.0.1:8000 --port 7860 --output "$EXP/videos"

# 4) 발표용 성공(>=200)/미달(<200) 사례 시드 탐색 (10000~10099, 20000~20199)
python demo_check.py --url http://127.0.0.1:8000 --output "$EXP"
```

### 제출 전 최종 검증 및 보고서 그래프·표 생성

터미널 A의 서버를 종료(`Ctrl+C`)한 뒤 실행한다:
```bash
python validate_results.py --output "$EXP"
python make_report.py
```

---

## 7. 실험 결과 및 상세 분석 (보고서·발표 핵심 내용)

모든 요약 표와 그래프 원본은 `artifacts/team/try5/report/` 폴더(`report_tables.md`, `tries_curve.png`, `learning_curve_5seeds.png`, `action_modes.png`, `noise.png`)에 저장되어 있다. 아래 분석에서 `(가설)`, `(추정)`으로 표시한 설명은 별도 실험으로 확인한 것이 아니라 결과를 보고 세운 해석이다.

### 7.1 베이스라인 결과 (2단계, 평가 seed 10000~10099, 각 100회)

| 정책 | 평균 return | 표준편차 (에피소드 간) | 성공률 (return $\ge 200$) |
|---|---|---|---|
| **랜덤 행동 (seed 12345)** | -172.7 | 105.0 | 0% |
| **무동작 (noop, 행동 0)** | -131.5 | 44.0 | 0% |

- 두 베이스라인 모두 100개 에피소드가 전부 `terminated`로 끝났고 `truncated`(시간 초과)는 0회였다(`baseline_random.csv`, `baseline_noop.csv`의 열을 집계해 확인한 값).
- 무동작 정책은 엔진을 쓰지 않고 자유낙하해 추락하는 에피소드가 대부분이라 평균 `-131.5`, 편차 `44.0`으로 비교적 일정하게 실패한 것으로 보인다(추정).
- 랜덤 정책은 무작위 엔진 분사로 매 스텝 연료 페널티(메인 -0.3, 측면 -0.03)가 누적되고 패드 밖 이탈이나 회전 추락이 섞여 평균 return(`-172.7`)이 더 낮고 표준편차(`105.0`)도 2배 이상 큰 것으로 보인다(추정).
- 4단계 PPO 모델(`argmax` 평균 `256.5`)과 비교하면 **랜덤 대비 +429.2점, 무동작 대비 +388.0점** 향상되었다.

### 7.2 대표 seed(`20237141`) 성능 개선 기록과 가설 분석 (3단계)

| try | 바꾼 설정 | 변경 이유 및 가설 (AI 제안 검토 근거) | `last_return` | 판정 |
|---|---|---|---|---|
| **try1** | RL Zoo 설정 그대로 | 기준 설정 (`update_epochs=4`, `anneal_lr=True`, `clip_vloss=True`, `vf_coef=0.5`) | 88.0 | 미달 |
| **try2** | `update_epochs`: 4 $\rightarrow$ 10 | `try1` 학습 곡선이 0~40만 스텝에서 빠르게 상승한 뒤 40만 스텝 이후 100 근처에서 정체됨. 수집된 배치(16,384 스텝)를 더 많이 반복 학습하도록 에폭 수만 증가 (가설) | 113.7 | 미달 |
| **try3** | `anneal_lr`: True $\rightarrow$ False | 학습 후반부에 학습률이 선형 감소해 0에 가까워지는 것이 후반 정체의 원인일 수 있다고 가정하고 `try2`에서 학습률 감소를 끔 (가설 $\rightarrow$ 96.5로 오히려 하락하여 기각) | 96.5 | 미달 |
| **try4** | `clip_vloss`: True $\rightarrow$ False | `try3` 변경을 되돌린 뒤(`try2` 기준), 보상 규모가 수백 점 단위로 큰 LunarLander 환경에서 이전 가치 추정치 대비 `±0.2`(`clip_coef`)로 가치 손실을 자르는 `clip_vloss=True`가 Critic 학습 속도를 제한한다는 가설 적용 | 186.2 | 미달 (큰 폭 개선) |
| **try5** | `vf_coef`: 0.5 $\rightarrow$ 1.0 | CleanRL 코드는 `v_loss = 0.5 * MSE`로 이미 0.5를 곱한 뒤 `loss = pg_loss - ent + v_loss * vf_coef`를 계산하므로 `vf_coef=0.5`일 때 실제 가중치가 `0.25`가 됨(`train_ppo.py`의 가치 손실 계산 부분에서 확인할 것). SB3 기본 비중(0.5)과 맞추기 위해 `vf_coef=1.0`으로 상향 (SB3 쪽 설명은 코드로 직접 대조하기 전까지 가설) | **263.7** | **통과 (최종 선택)** |

- **AI 제안 중 적용하지 않은 항목과 이유**:
  1. `learning_rate`, `ent_coef` 조정: 후보로 논의했으나 한 `try`에 설정 하나만 바꿔야 어떤 변수가 성능 변화를 만들었는지 구분할 수 있어 보류했다. (AI는 try2~try5에서 매번 설정 하나씩(`update_epochs`, `anneal_lr`, `clip_vloss`, `vf_coef`)을 제안했다.)
  2. 관측 정규화(`NormalizeObservation` 래퍼): 학습 시 관측 정규화를 적용하면 4~8단계 평가 및 FastAPI 서버 배포 시에도 동일한 정규화 통계량(mean/var)을 저장·로드해야 하므로 파이프라인 복잡도와 오류 위험이 커서 제외했다(과제 슬라이드에서도 비추천).

### 7.3 두 가지 행동 선택 방식 비교: `argmax` vs 확률적 선택 (4단계)

| 학습 seed | `argmax` 평균 | `argmax` 표준편차 | 확률적(`sample`) 평균 | 확률적(`sample`) 표준편차 | 3단계 학습 `last_return` |
|---|---|---|---|---|---|
| `20237141` | 254.4 | 21.4 | 263.7 | 44.5 | 263.7 |
| `30237141` | 260.9 | 20.9 | 260.9 | 44.1 | 263.3 |
| `40237141` | 252.3 | 25.4 | 146.3 | 40.3 | 140.6 |
| **`50237141` (Best)** | **261.8** | **21.0** | 219.7 | 57.4 | 186.9 |
| `60237141` | 253.4 | 21.8 | 261.7 | 42.4 | 252.0 |
| **5개 평균 ± 표준편차** | **`256.5 ± 4.0`** | **`22.1`** | **`230.4 ± 45.2`** | **`45.7`** | `221.3` |

- **표 읽는 법**: 각 모델의 표준편차 열은 에피소드 100개 사이의 표준편차(5개 평균: `argmax 22.1`, `sample 45.7`)이고, 맨 아래 요약 행의 `±` 값은 5개 모델 평균들 사이의 표준편차(`np.std`, $n=5$로 나눔: `argmax ±4.0`, `sample ±45.2`)이다.
- **200 미만 에피소드 종료 원인 비교 (`terminated` vs `truncated`)**:
  - **`argmax` (전체 500회 중 2회 미달)**: `terminated 0회`(추락 없음), `truncated 2회`(시간 초과). 5개 모델 모두 252~261점대로 균일하고 안정적인 착륙 성능을 보인다.
  - **`sample` (전체 500회 중 155회 미달)**: `terminated 20회`, **`truncated 135회`**. 즉 200 미만 사례의 **87.1%(135/155)가 추락이 아니라 1,000스텝 시간 초과(`truncated`)**였다.
  - **모델별 200 미만 횟수 (미달 횟수, `terminated`/`truncated`)**: `sample`은 `20237141` 4회(4/0), `30237141` 5회(2/3), `40237141` 99회(10/89), `50237141` 44회(1/43), `60237141` 3회(3/0)였고, `argmax`는 `40237141`만 2회(0/2)이며 나머지는 0회였다(평가 CSV 집계로 확인한 값).
- **두 방식 차이의 이유 분석**:
  - 학습 중에는 탐색을 위해 Softmax 확률 분포에서 행동을 뽑는 확률적 선택(`sample`)을 쓰지만, 수렴 후 평가에서 `sample`을 쓰면 착륙 직전 미세 제어 구간에서 차선책 행동(불필요한 메인/측면 엔진 분사)이 간헐적으로 샘플링될 수 있다. 이로 인해 착륙선이 자세를 흔들며 패드 위에서 떠 있다가 연료 페널티만 누적되고 1,000스텝 시간 초과(`truncated`)로 끝나는 것으로 보인다(가설).
  - 3단계 학습 로그에서 `last_return`이 낮았던 `40237141`(140.6)과 `50237141`(186.9)은 4단계 `sample` 평가에서도 각각 `146.3`, `219.7`로 낮게 나왔다(학습 방식과 같은 확률적 선택이므로 비슷한 경향이 나온 것으로 해석, 가설). 같은 모델에 `argmax`를 적용하면 각각 `252.3`, `261.8`로 점수가 올랐다(관측된 결과). 가장 큰 logit 기준의 행동 선택은 비교적 잘 학습되어 있었던 것으로 보이지만(가설), logits 격차 등을 직접 측정해 확인하지는 않았다.
  - 따라서 실제 서비스 배포(`serve.py`)에서 `argmax`를 사용하는 것이 평균 return(+26.1점)과 안정성(모델 간 편차 45.2 $\rightarrow$ 4.0) 양면에서 유리하다.

### 7.4 관측 노이즈($\sigma$)에 따른 성능 변화 (5단계)

| 노이즈 크기 ($\sigma$) | 5개 모델 평균 return | 5개 모델 간 표준편차 ($\pm$) | $\sigma=0$ 대비 변화 | `20237141` | `30237141` | `40237141` | `50237141` | `60237141` |
|---|---|---|---|---|---|---|---|---|
| **0.0** | **256.5** | **4.0** | **+0.0** | 254.4 | 260.9 | 252.3 | 261.8 | 253.4 |
| **0.01** | **256.0** | **4.8** | **-0.6** | 254.0 | 260.4 | 250.3 | 262.8 | 252.5 |
| **0.05** | **259.4** | **8.6** | **+2.8** | 263.0 | 267.7 | 243.0 | 264.1 | 258.9 |
| **0.1** | **228.5** | **45.4** | **-28.1** | 258.3 | 266.7 | **140.3** | 240.2 | 237.0 |
| **0.2** | **110.9** | **23.4** | **-145.7** | 122.2 | 102.2 | 89.4 | 89.4 | 151.1 |

- **$\sigma=0.0$ 점검**: 4단계 `argmax` 평균(`256.5 ± 4.0`)과 정확히 일치한다.
- **$\sigma$별 200 미만 에피소드 수** (5개 모델 × 100회 = 500회 중, `terminated`/`truncated`): $\sigma=0.0$은 2회(0/2), $\sigma=0.01$은 7회(3/4), $\sigma=0.05$는 21회(8/13), $\sigma=0.1$은 142회(43/99), $\sigma=0.2$는 421회(224/197)이다(노이즈 CSV 집계로 확인한 값). `terminated`는 추락뿐 아니라 착륙 종료도 포함하며, 여기서는 200 미만인 에피소드만 센 값이다.
- **$\sigma = 0.01 \sim 0.05$ 구간**: 변화량(-0.6, +2.8)이 모델 간 표준편차(4.8, 8.6) 안쪽이므로 유의미한 성능 향상이나 하락이 없는 구간으로 해석해야 한다(+2.8을 "향상"으로 해석하지 않는다).
- **$\sigma = 0.1$ 구간 (왜 모델 간 표준편차가 `45.4`로 급증했는가?)**:
  - 나머지 4개 모델은 $\sigma=0.1$에서도 `237.0 ~ 266.7`점을 유지했지만, **`seed 40237141` 모델 하나만 `252.3` $\rightarrow$ `140.3`으로 급락**했기 때문이다(위 표에서 확인되는 사실).
  - $\sigma=0.1$에서 200 미만 에피소드는 모델별로 `20237141` 8회(`terminated` 3, `truncated` 5), `30237141` 3회(1, 2), `40237141` 99회(16, 83), `50237141` 18회(10, 8), `60237141` 14회(13, 1)였다(합계 142회, `results/noise/*_sigma0.1.csv` 집계로 확인한 값). `40237141`은 100회 중 99회가 200 미만이었다.
  - `40237141`은 4단계 확률적 선택에서도 5개 중 가장 낮았다(`146.3`). 두 현상이 같은 원인(예: 행동 logits 간 격차가 작아 관측 잡음이나 샘플링에 의해 행동이 쉽게 바뀜)일 가능성이 있지만 logits 격차를 측정해 검증하지는 않았다(가설).
- **$\sigma = 0.2$ 구간**: 5개 모델 모두 `89.4 ~ 151.1`(평균 `110.9`, -145.7점 하락)로 성능이 절반 이하로 떨어진다. 잡음이 관측 8개 모두에 더해지는 설계라 다리 접촉(7, 8번, 원래 0 또는 1) 값과 각도·속도 값이 크게 흔들려 착륙 판단이 어려워졌을 수 있다(가설, 접촉 센서만 따로 분리한 실험은 하지 않음).

### 7.5 ONNX 변환 검증, 추론 지연 및 크기 비교 (6~7단계)

- **변환 대상**: Best 모델(`seed 50237141`, `models/best.pt`)의 Actor 신경망 (`models/policy.onnx`, `sha256`: `2885ac92b608741a9de7b8b5ddfe34ed0a2cc72e0e15b835e144de51210dc135`, `results/export.json`의 `policy_sha256`과 일치함을 확인)

| 비교 항목 | PyTorch (`actor.pt`) | ONNX (`policy.onnx`) | 비고 |
|---|---|---|---|
| **Actor 파일 크기 (KiB)** | **22.1** (`22.12 KiB`) | **25.9** (`25.94 KiB`) | Critic 제외한 Actor 순수 파일 크기 비교 |
| **추론 지연 평균 · p95 (ms)** | **0.0232 · 0.0326** | **0.0058 · 0.0072** | 동일 PC, CPU 스레드 1개, warmup 100회 후 1,000회 측정 |
| **행동 일치율 (관측 10,000개)** | 기준 | **1.0 (100% 일치)** | seed 0으로 수집한 실제 관측 10,000개에서 `argmax` 비교 |
| **최대 절대 오차 (logits)** | 기준 | **`2.86e-06`** | float32 연산 순서 차이에 의한 미세 오차 수준 |
| **평균 절대 오차 (logits)** | 기준 | **`1.14e-07`** | 모든 관측에서 유한성(`finite: true`) 확인 |

- **왜 ONNX가 파일 크기는 조금 더 큰데 추론 속도는 약 4배 빠른가?**:
  - **크기 이유**: PyTorch의 `actor.pt`는 순수 가중치 텐서(`state_dict`)만 저장하지만, `policy.onnx`(26,558 바이트)는 가중치 외에 **계산 그래프 구조(연산자 노드 연결 정보), 입출력 텐서 명세(`obs`, `logits`), opset 18 메타데이터**를 함께 저장한다. 신경망 자체가 `8-64-64-4`로 매우 작기 때문에 그래프 메타데이터 비중(약 3.8 KiB)이 더해져 ONNX 파일이 소폭 더 크게 나타난다(추정).
  - **속도 이유**: 관측 1개(`batch=1`)씩 1스레드로 추론할 때, PyTorch는 매 호출마다 Python 레벨의 텐서 생성(`torch.tensor`)과 동적 디스패치 오버헤드가 상대적으로 큰 반면, ONNX Runtime은 고정된 정적 그래프를 C++ 런타임에서 바로 실행하여 함수 호출 오버헤드가 줄어들기 때문에 평균 지연이 약 4.0배(p95 기준 4.5배) 빨라졌다(추정).

### 7.6 FastAPI 서버 배포 및 Gradio 시연 사례 탐색 (8단계)

- FastAPI 서버(`serve.py`)의 `/model` 엔드포인트가 반환하는 `sha256` 해시가 `results/export.json`의 `policy_sha256`과 일치함을 확인하여, 배포된 모델이 `best.pt`(`50237141`)에서 변환된 `policy.onnx`임을 검증했다.
- `/act` 엔드포인트에 길이 8의 정상 배열 입력 시 `0~3` 행동을 반환(HTTP 200)하고, 길이 3 배열 등 잘못된 입력 시 Pydantic 검증으로 추론 실행 전에 `HTTP 422`를 반환함을 확인했다.
- Gradio(`app.py`)에서 `seed 10000`을 두 번 실행해 각각 고유한 MP4 파일이 생성되고, return `252.7`이 4단계 PyTorch 평가(`eval_50237141_argmax.csv`의 `252.687`)와 일치함을 확인했다.
- **시연 사례 탐색(`demo_check.py`)**: 평가 seed `10000~10099`(100개)와 추가 탐색 seed `20000~20199`(200개) **총 300개 에피소드를 순서대로 모두 실행했으나, `return < 200`인 미달 사례가 단 1건도 발생하지 않았다(300회 모두 $\ge 200$ 성공)**. 따라서 과제 규정("범위를 다 확인해도 한 종류뿐이면 확보한 영상과 탐색 표·설명을 제출")에 따라 성공 사례 영상(`seed 10000`, return `252.7`)과 300개 전체 탐색 기록 표(`results/demo_cases.csv`)를 함께 제출한다. 이 추가 탐색 결과는 4단계 평가 평균에 섞지 않았다. 300개 에피소드의 return은 평균 261.2, 최소 211.4, 최대 302.9이며, best 모델 선택에 쓰이지 않은 `20000~20199` 구간의 평균은 261.0이다(`demo_cases.csv` 집계로 확인한 값).

---

## 8. 로그 매핑 표 (`try` 폴더 → `seed` → `runs/` → 모델 파일)

모든 경로는 `artifacts/team/` 기준이다. 학습 시간(분)은 로그 폴더명의 시작 Unix 시각과 체크포인트 파일 저장 시각의 차이로 계산한 실제 측정값이다. TensorBoard 로그는 `tensorboard --logdir artifacts/team`으로 확인할 수 있다.

| try | 학습 seed | `runs/` TensorBoard 로그 폴더 | `models/` 체크포인트 파일 | `last_return` | 학습 시간(분) |
|---|---|---|---|---|---|
| **try1** | `20237141` | `try1/runs/LunarLander-v3__train_ppo__20237141__1794725280/` | `try1/models/LunarLander-v3__train_ppo__20237141__1794725280.pt` | 88.0 | 2.7 |
| **try2** | `20237141` | `try2/runs/LunarLander-v3__train_ppo__20237141__1794727424/` | `try2/models/LunarLander-v3__train_ppo__20237141__1794727424.pt` | 113.7 | 5.7 |
| **try3** | `20237141` | `try3/runs/LunarLander-v3__train_ppo__20237141__1794728176/` | `try3/models/LunarLander-v3__train_ppo__20237141__1794728176.pt` | 96.5 | 5.7 |
| **try4** | `20237141` | `try4/runs/LunarLander-v3__train_ppo__20237141__1794728686/` | `try4/models/LunarLander-v3__train_ppo__20237141__1794728686.pt` | 186.2 | 9.8 |
| **try5** | `20237141` | `try5/runs/LunarLander-v3__train_ppo__20237141__1794731637/` | `try5/models/LunarLander-v3__train_ppo__20237141__1794731637.pt` | 263.7 | 15.1 |
| **try5** | `30237141` | `try5/runs/LunarLander-v3__train_ppo__30237141__1794733206/` | `try5/models/LunarLander-v3__train_ppo__30237141__1794733206.pt` | 263.3 | 11.1 |
| **try5** | `40237141` | `try5/runs/LunarLander-v3__train_ppo__40237141__1794733874/` | `try5/models/LunarLander-v3__train_ppo__40237141__1794733874.pt` | 140.6 | 8.7 |
| **try5** | `50237141` | `try5/runs/LunarLander-v3__train_ppo__50237141__1794734398/` | `try5/models/LunarLander-v3__train_ppo__50237141__1794734398.pt` | 186.9 | 5.3 |
| **try5** | `60237141` | `try5/runs/LunarLander-v3__train_ppo__60237141__1794734715/` | `try5/models/LunarLander-v3__train_ppo__60237141__1794734715.pt` | 252.0 | 5.1 |

---

## 9. 실험의 한계와 주의사항 (질의응답 대비)

1. **단일 대표 seed 기반 튜닝의 한계**: `try1`~`try5` 개선 과정은 과제 규정에 따라 대표 seed(`20237141`) 하나로만 실험했다. 강화학습은 seed별 분산이 크므로(실제로 `try5`에서도 5개 seed의 `last_return`이 `140.6 ~ 263.7`로 분포함), `try2`(113.7)와 `try3`(96.5) 같은 작은 차이나 `clip_vloss`의 효과를 단일 seed 결과만으로 일반화하기는 어렵고 다중 seed 반복 검증이 필요하다.
2. **RL Zoo 원본과의 차이**: RL Zoo의 `ppo.yml`에 명시된 하이퍼파라미터만 가져왔으며, 명시되지 않은 값(학습률 `2.5e-4` 선형 감소, `max_grad_norm=0.5`, Orthogonal 가중치 초기화 등)은 CleanRL 기본값을 그대로 사용했으므로 Stable-Baselines3(RL Zoo) 원본 실험과 완전히 동일하지는 않다.
3. **Best 모델 선택 편향(Selection Bias)**: 평가 seed `10000~10099`(100회) 점수를 기준으로 `best.pt`(`50237141`, 평균 `261.8`)를 골랐기 때문에, 해당 100개 에피소드에 대한 점수는 모델의 일반화 성능보다 약간 높게 추정(낙관적 편향)되었을 수 있다. 시연 사례 탐색(`demo_check.py`)의 처음 100개 seed도 같은 구간이다. 다만 선택에 쓰이지 않은 `20000~20199` 200개 seed에서도 평균 261.0으로 모두 200 이상이었으므로 이 편향이 결과를 크게 바꾸지는 않았을 것으로 보인다(추정).
4. **이산 접촉 센서에 대한 가우시안 노이즈 주입**: 5단계에서 관측 8개 전체에 연속적인 정규분포 잡음을 더했는데, 7·8번 관측값(다리 접촉 여부)은 원래 `0` 또는 `1`의 이진값이다. 실제 환경에서는 접촉 센서에 가우시안 잡음보다 비트 플립(0 $\leftrightarrow$ 1) 형태의 오류 모델을 적용하는 것이 더 현실적인 개선 방향이다.
5. **짧은 학습 점검 제외**: `artifacts/smoke/`는 코드 동작 확인용(16,384 스텝)이며 본 실험 결과 및 최종 제출 압축 파일에서 제외한다.
6. **원인 설명은 가설**: 7장의 `(가설)`, `(추정)` 표시 문장(두 행동 방식 차이의 이유, 노이즈에서 `40237141`이 약했던 이유, `clip_vloss`와 `vf_coef`의 효과, ONNX 크기·속도 차이의 이유)은 결과를 보고 세운 해석이며, logits 격차 측정이나 ablation 같은 추가 실험으로 검증하지 않았다.

---

## 10. 핵심 용어 정리

- **return / 성공률**: 한 에피소드 동안 받은 보상의 총합 / `return >= 200`인 에피소드의 비율.
- **`last_return`**: 총 100만 스텝 학습 중 마지막 10만 스텝(900,000 ~ 1,000,000 스텝) 구간에 종료된 에피소드들의 평균 return (3단계 대표 seed 통과 판정 기준값).
- **Actor / Critic**: 관측(8개)을 받아 행동 4개의 logits를 출력하는 정책 신경망 / 상태의 가치(스칼라 1개)를 추정하는 가치 신경망. 추론 및 ONNX 배포에는 Actor만 사용한다.
- **`argmax` / 확률적 선택(`sample`)**: logits가 가장 큰 행동을 결정적으로 선택하는 방식(배포 방식) / logits에 Softmax를 취한 확률 분포에서 행동을 샘플링하는 방식(학습 방식).
- **$\sigma$ (노이즈 크기)**: 매 스텝 관측값 8개 각각에 더하는 표준정규분포 난수의 표준편차.
- **p95 지연 시간**: 1,000회 추론 시간 측정값 중 상위 5% 느린 경계값(95번째 백분위수, ms).

---

## 11. 팀원 및 역할 분담

- **팀원 및 역할 분담**:
  - **정다형 (20231994)**
  - **백연우 (20237141)**
  - **박재우 (20232020)**
- **AI 대화 증빙**:
  - 보고서 부록에 사용한 생성형 AI 대화 전체 내보내기 파일(PDF/텍스트) 및 공유 링크 첨부
