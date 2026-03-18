# lane_agent_project

MMS LAS 데이터에서 차선을 추적하는 Python 기반 룰형 에이전트 예제입니다.

## 입력
- LAS 파일
- 시작점 P0(x, y, z)
- 두 번째 점 P1(x, y, z)
- config.yaml

## 출력
- CSV(x, y, z)
- 선택: debug json

## 설치
```bash
pip install laspy numpy
```

## 실행
```bash
python -m lane_agent.cli \
  --las data/sample.las \
  --p0 314383.1 3899679.2 75.2 \
  --p1 314384.0 3899679.0 75.2 \
  --config config.yaml \
  --output lane_points.csv
```

## 구성
- `lane_agent/cli.py` : 실행 엔트리
- `lane_agent/config.py` : 설정 로드
- `lane_agent/las_io.py` : LAS xyz/intensity 로드
- `lane_agent/grid.py` : spatial grid
- `lane_agent/scoring.py` : 후보 점수 계산
- `lane_agent/agent.py` : 차선 추적 에이전트
- `lane_agent/csv_io.py` : 결과 저장

## 참고
현재 기본 파라미터는 실제 데이터 기반 튜닝 전의 시작값입니다. 따라서 `step_m`, `max_gap_m`, `search_half_width_m`, `min_score`는 실제 LAS에 맞춰 조정이 필요할 수 있습니다.
