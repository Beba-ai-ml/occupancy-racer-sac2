# Curriculum Learning -- strategia na finalny 10-godzinny trening SAC

## Streszczenie

Badania nad curriculum learning w RL nawigacyjnym jednoznacznie wskazują: stopniowe zwiększanie trudności środowiska przyspiesza konwergencję i daje lepszą końcową wydajność niż trening od razu na pełnym zbiorze map. Kluczowe jest jednak **mieszanie łatwych zadań z trudnymi** na każdym etapie -- czysto sekwencyjne curriculum (łatwe -> trudne bez powrotu) prowadzi do katastrofalnego zapominania. Uniform sampling ze wszystkich poziomów trudności jest "zaskakująco silnym baselinem" (Narvekar et al., JMLR 2020).

## Kiedy zwiększać trudność?

Z literatury wyłaniają się dwa podejścia:

**1. Próg wydajności (preferowane).** Badanie curriculum UAV w tunelach stosowało 3-poziomowy system z progiem sukcesu **80-85%** do awansu na kolejny level. Zbyt wyśrubowane progi (>90%) drastycznie wydłużają czas treningu na danym etapie bez proporcjonalnej poprawy ([arXiv:2512.10934](https://arxiv.org/html/2512.10934v1)). W kontekście projektu odpowiedni próg to np. średni dystans z ostatnich 50 epizodów > 80% dotychczasowego rekordu fazy.

**2. Automatic Domain Randomization (ADR).** OpenAI's ADR startuje z wąskim zakresem parametrów i poszerza go automatycznie, gdy policy trzyma wynik powyżej progu. Stosunek eksploatacji trudnych scenariuszy do eksploracji nowych: **80/20** (parametr Bernoulli D=0.8). Scenariusze dodawane do bufora tylko gdy ich "learning potential" przekracza minimum -- eliminuje ręczne heurystyki ([arXiv:2505.08264](https://arxiv.org/html/2505.08264)).

**Wniosek:** Progresja oparta na metryce adaptuje się do tempa nauki agenta. Stały harmonogram epizodowy (np. "co 2000 epizodów") ryzykuje albo za wczesnym przeskokiem, albo marnowaniem czasu na opanowany już poziom.

## Proponowany harmonogram na 10 godzin

Na podstawie F1Tenth racing paper (24 tracki, TD3, [arXiv:2510.26040](https://arxiv.org/html/2510.26040)), curriculum nawigacji UGV z LiDARem ([ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2215098625002022)) i doświadczeń z ADR:

| Faza | Czas | Mapy / trudność | DR zakresy | Kryterium awansu |
|------|------|------------------|------------|-------------------|
| **1. Fundament** | ~2.5h (25%) | 3-5 otwartych map (K-serie, szerokie korytarze). Brak przeszkód, brak opponenta. | Węższe (np. accel_scale [0.85, 1.15]) | Średni dystans >80% rekordu fazy |
| **2. Rozszerzenie** | ~3h (30%) | Pool 10-15 map (dodać J-serie, węższe przejścia). Przeszkody statyczne od połowy fazy. | Docelowe zakresy (jak w configu) | Dystans stabilny przez 200+ epizodów |
| **3. Komplikacja** | ~2.5h (25%) | Pool 25-30 map (wąskie drzwi, Dom/Hala). Przeszkody dynamiczne + opponent bot. | Pełne zakresy DR | Success rate >70% na trudnych mapach |
| **4. Polerowanie** | ~2h (20%) | Pełny pool 40+ map. Ważona rotacja: 40% trudne, 40% średnie, 20% łatwe. | Pełne DR + ewentualnie Active DR | Monitorowanie stabilności, brak regresji |

Podział 25/30/25/20 odzwierciedla badania: fundament wymaga solidnej bazy, największa część czasu na rozszerzanie kompetencji, a faza polerowania celowo krótsza bo buduje na solidnym fundamencie.

## Rotacja map i pool

F1Tenth paper stosował **24 tracki** (6 bazowych x 4 szerokości). Twój zestaw 51 map jest bogatszy -- kluczowe to nie wprowadzać ich wszystkich naraz.

- **Faza 1-2:** `map_switch_every: 15` (obecna wartość) jest dobra -- nowe mapy dodawane stopniowo.
- **Faza 3-4:** Ważona rotacja inspirowana ADR (80% eksploatacja map, na których agent się jeszcze uczy, 20% eksploracja łatwych dla utrzymania bazy).
- **Lekcja z projektu:** `stratified_sampling: true` zniszczyło naturalny curriculum w Mapa_1_4 (131m vs 284m bez). Lepiej sterować trudnością przez dobór map niż przez manipulację replay buffera.

## Wall-following a gap-finding -- razem czy osobno?

Badania sugerują **fazowe podejście z nakładaniem się umiejętności**, a nie sztywną separację:

1. Curriculum od unikania kolizji -> nawigacja korytarzami -> omijanie przeszkód dało **93% success rate** w środowiskach fabrycznych ([ScienceDirect - UGV navigation](https://www.sciencedirect.com/science/article/pii/S2215098625002022)).
2. Gap-finding nie powinien być oddzielnym etapem treningowym -- powinien wynikać naturalnie z **coraz węższych przejść w mapach**. F1Tenth paper celowo unikał dedykowanych nagród za overtaking bo prowadziło to do reward hackingu.
3. Obecny reward (forward_speed_weight=0.6, side_penalty=0.02, min_clear_penalty=0.04) już koduje "jedź szybko, ale zachowaj dystans" -- to wystarczająca baza. Agent nauczy się gap-finding przez to, że wąskie mapy wymuszają precyzyjne manewrowanie.

**Wniosek:** Nie trenować osobno wall-following i gap-finding. Zamiast tego -- progresja szerokości korytarzy w mapach (szerokie -> wąskie) naturalnie wymusi obie umiejętności.

## Domain Randomization -- harmonogram

Badania ADR ([OpenAI Rubik's Cube](https://arxiv.org/abs/1910.07113), [Active DR](http://proceedings.mlr.press/v100/mehta20a/mehta20a.pdf)) wskazują, że **stopniowe poszerzanie zakresów DR** jest efektywniejsze niż uniform:

- **Faza 1:** DR włączone, ale z węższymi zakresami (accel_scale [0.85, 1.15], friction [0.85, 1.15]). Agent uczy się podstaw bez ekstremalnych perturbacji.
- **Faza 2-3:** Stopniowe poszerzanie do docelowych wartości z configu ([0.7, 1.3]).
- **Faza 4:** Pełne zakresy. Active DR próbkuje więcej trudnych perturbacji (zamiast uniform).

Ważne: sensor noise (lidar_noise_std, lidar_drop_prob) powinien być obecny od początku -- to "stały szum" rzeczywistości, a nie element trudności.

## Kluczowe ryzyki i zabezpieczenia

| Ryzyko | Zabezpieczenie |
|--------|----------------|
| Catastrophic forgetting przy przejściu między fazami | Zawsze mieszać min. 20% łatwych map w poolu ([Zaremba & Sutskever](https://lilianweng.github.io/posts/2020-01-29-curriculum-rl/)) |
| Alpha collapse przy nagłym skoku trudności | Monitorować alpha przy zmianach faz. alpha_min=0.05 i alpha_max=0.3 dają bufor. |
| Q-divergence na nowych mapach | grad_clip=0.5 i learn_after=5000 dają margines bezpieczeństwa |
| Premature advancement | Wymóg stabilności (200+ epizodów) przed awansem, nie jednorazowy spike |

---

*Źródła:*
- [Lil'Log - Curriculum for Reinforcement Learning](https://lilianweng.github.io/posts/2020-01-29-curriculum-rl/)
- [JMLR - Curriculum Learning for RL Domains](https://jmlr.org/papers/volume21/20-212/20-212.pdf)
- [F1Tenth Overtaking with RL (arXiv:2510.26040)](https://arxiv.org/html/2510.26040)
- [UAV Curriculum Navigation (arXiv:2512.10934)](https://arxiv.org/html/2512.10934v1)
- [Automatic Curriculum for Driving (arXiv:2505.08264)](https://arxiv.org/html/2505.08264)
- [UGV Navigation with Curriculum (ScienceDirect)](https://www.sciencedirect.com/science/article/pii/S2215098625002022)
- [OpenAI ADR - Rubik's Cube (arXiv:1910.07113)](https://arxiv.org/abs/1910.07113)
- [Active Domain Randomization (PMLR)](http://proceedings.mlr.press/v100/mehta20a/mehta20a.pdf)
- [Stable Baselines3 RL Tips](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html)
