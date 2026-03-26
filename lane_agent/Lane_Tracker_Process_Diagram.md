# Lane Tracker Process Diagram

```mermaid
flowchart TD
    A0([Start]) --> A1[CLI 인자 파싱<br/>--las --p0 --p1 --config --output]
    A1 --> A2[Config 로드<br/>key:value 파싱]
    A2 --> A3[LAS 로드<br/>xyz + intensity]
    A3 --> A4[LaneTrackerAgent 생성]
    A4 --> A5[SpatialGrid 구축<br/>cell_size=grid_size_m]
    A5 --> A6[초기 상태 세팅<br/>cur=p0, cur_dir=unit(p1-p0)]

    A6 --> B1[p0 주변 반경 seed_profile_radius_m 조회]
    B1 --> B2[SeedProfile 추정<br/>target=I q90, background=I q35, z_ref=Z median]
    B2 --> B3[cur.z 보정<br/>_fit_center_z]

    B3 --> C0[루프 초기화<br/>points=[cur], scores=[1.0], traveled=0, gap_accum=0]
    C0 --> C1{traveled < max_track_length_m ?}

    C1 -->|No| X0[stop_reason=max_length_reached]
    C1 -->|Yes| C2[방향 예측 _predict_direction<br/>최근 direction_history_points 가중 합]

    C2 --> D0[후보 생성 _candidate_centers]
    D0 --> D1[기본 후보<br/>forward_fracs 0.85~1.15<br/>x lateral_offsets -half~+half]
    D0 --> D2[곡선 후보<br/>±max_heading_change_deg 내 회전 7샘플]
    D1 --> D3[후보 반복 시작]
    D2 --> D3

    D3 --> E1[후보 중심 c_xy 주변 반경 search_radius_m 조회]
    E1 --> E2[c_z 적합 _fit_center_z<br/>center3=(x,y,z)]

    E2 --> E3{하드 게이트 통과?}
    E3 -->|No| E4[reject: hard_gate 기록<br/>이유: lateral hard limit 또는 hard max z step]
    E4 --> E15{다음 후보 있음?}
    E15 -->|Yes| D3
    E15 -->|No| F0

    E3 -->|Yes| E5[score_candidate 계산]
    E5 --> E6[intensity_term]
    E5 --> E7[contrast_term]
    E5 --> E8[continuity_term]
    E5 --> E9[straight_term]
    E5 --> E10[z_term]
    E5 --> E11[z_step_term]
    E5 --> E12[center_term]
    E6 --> E13[가중합 + 곱 패널티로 score 산출]
    E7 --> E13
    E8 --> E13
    E9 --> E13
    E10 --> E13
    E11 --> E13
    E12 --> E13

    E13 --> E14[lane_loyalty 적용]
    E14 --> E16{best_score 갱신?}
    E16 -->|Yes| E17[best_center, best_dir 갱신]
    E16 -->|No| E18[기존 best 유지]
    E17 --> E15
    E18 --> E15

    F0{best_center 존재?} -->|No| X1[stop_reason=no_candidate]
    F0 -->|Yes| F1[best_center 1차 보정 _refine_center_xy]
    F1 --> F2[best_center 2차 보정 _refine_centerline_cross_section]
    F2 --> F3[best_center z 재적합 _fit_center_z]

    F3 --> G0{best_score >= min_score ?}

    G0 -->|No| G1[gap_accum += step_m, gap_steps++]
    G1 --> G2{gap_accum > max_gap_m ?}
    G2 -->|Yes| X2[stop_reason=score_below_threshold]
    G2 -->|No| G3[cur = best_center]
    G3 --> G4[cur_dir = pred_dir 또는 best_dir]
    G4 --> G5[traveled += step_m]
    G5 --> G6[debug step 기록(mode=gap_bridge)]
    G6 --> C1

    G0 -->|Yes| H1[gap_accum 리셋]
    H1 --> H2[cur=best_center, cur_dir=best_dir]
    H2 --> H3{프로파일 업데이트 조건 충족?}
    H3 -->|Yes| H4[_refresh_seed_profile + blend(alpha)]
    H3 -->|No| H5[프로파일 유지]
    H4 --> H6[points append, scores append]
    H5 --> H6
    H6 --> H7[traveled += 실제 XY 이동거리]
    H7 --> H8[debug step 기록(mode=accept)]
    H8 --> C1

    X0 --> P0[raw_arr 생성]
    X1 --> P0
    X2 --> P0

    P0 --> P1{enable_post_correction && point>=7 ?}
    P1 -->|Yes| P2[_post_correct_detours]
    P1 -->|No| P3[skip]
    P2 --> P4{window>=3 && point>=3 ?}
    P3 --> P4
    P4 -->|Yes| P5[_smooth 이동평균]
    P4 -->|No| P6[skip]
    P5 --> P7[최종 arr]
    P6 --> P7

    P7 --> P8{save_debug_json ?}
    P8 -->|Yes| P9[debug JSON 저장]
    P8 -->|No| P10[skip]

    P9 --> Q1[CSV 저장 2종<br/>raw / post-corrected]
    P10 --> Q1
    Q1 --> Q2([End])
```