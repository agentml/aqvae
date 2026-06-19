# docs 인덱스

> 문서는 `mmdd_` 접두사(파일 생성일) 정렬. 새 문서도 생성일 접두사를 붙인다.
> (`conversation_raw.md`는 트랜스크립트 추출 생성물 — 접두사 없음, gitignore.)

## 설계 (Design)
- [`0617_proposal.md`](0617_proposal.md) — 논문형 제안(AQ: 미분가능 이산 표현, attractor+IFT).
- [`0617_design.md`](0617_design.md) — 실무 설계서(결정·코드구조·로드맵·미결).
- [`0618_categorical_aqvae_design.md`](0618_categorical_aqvae_design.md) — Categorical AQ-VAE(변분: w* posterior + 학습 prior + KL).
- [`0618_gaussian_aqvae_design.md`](0618_gaussian_aqvae_design.md) — Gaussian-AQ-VAE(context에 Gaussian 노이즈, reparam).
- [`0618_local_learning.md`](0618_local_learning.md) — 국소(생물학적) 학습 설계(경쟁Hebbian·predictive coding·EqProp).

## 결과 (Results)
- [`0618_results.md`](0618_results.md) — Phase F 평가(재구성·disentanglement·의미분석·진단·변분 ablation 비교표).

## 분석 (Analysis)
- [`0617_collapse_analysis.md`](0617_collapse_analysis.md) — representation collapse 진단·기전·해결(train soft / anneal to discrete).
- [`0618_aq_vs_gumbel.md`](0618_aq_vs_gumbel.md) — AQ vs (ST-)Gumbel-Softmax vs STE 관계.
- [`0618_neuroscience_significance.md`](0618_neuroscience_significance.md) — 계산신경과학적 의미(attractor·associative memory·개념형성).

## 메타·추적 (Meta)
- [`0618_todo.md`](0618_todo.md) — 모델·평가 개선 추적(우선순위·출처).
- [`0618_naming.md`](0618_naming.md) — 모델명 AQ 결정 + 논문 제목 후보.
- [`0618_project_log.md`](0618_project_log.md) — 전체 작업 로그(시간순 요약).
- [`0618_conversation_log.md`](0618_conversation_log.md) — 대화 로그(재구성).

## 읽는 순서 (추천)
신규 합류 시: `0617_proposal` → `0617_collapse_analysis` → `0618_results` → (관심사) 변분(`categorical/gaussian_aqvae`)·신경과학(`neuroscience`/`local_learning`).
