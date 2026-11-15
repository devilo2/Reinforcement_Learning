import argparse
import requests

URL = "http://127.0.0.1:8000"
S = requests.Session()


def act(obs):
    body = {"obs": [float(v) for v in obs]}
    r = S.post(URL + "/act", json=body, timeout=5)
    r.raise_for_status()
    a = r.json()["action"]
    if a not in (0, 1, 2, 3):
        raise ValueError(a)
    return a


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=URL)
    URL = ap.parse_args().url
    print("정상 입력 ->", act([0, 0, 0, 0, 0, 0, 1, 1]))
    try:
        act([0, 0, 0])
    except requests.HTTPError as e:
        print("잘못된 입력 ->", e.response.status_code)
        
