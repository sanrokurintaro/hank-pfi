# hank_pfi

ZLB(ゼロ金利制約)のような occasionally-binding constraint を含む HANK モデルを、
**Policy Function Iteration(Howard改善)+ 大域グリッド**で解くためのパッケージ。

設計方針は「 QuantEcon `DiscreteDP` 中核型」。家計の動的計画問題は自前でVFI/PFIループを
書かず、QuantEcon.py の `DiscreteDP.solve(method='policy_iteration')`(= Howard改善そのもの)に
委譲し、ZLBの遷移パスは Guerrieri & Iacoviello (2015) の OccBin と同じ発想の
「regime-switching guess-and-verify」で解く。

## 離散選択(PFI/DiscreteDP) vs EGM

`master`ブランチ(このブランチ)は、家計ブロックのSequence-Space Jacobianを**補間なしの
離散選択PFI**(`household.py`、DiscreteDP)から計算している。これには、微小な価格変化では
消費が反応しない「階段関数」問題や、賃金Jacobianの符号が反転するほどのノイズが発生する
という既知の限界がある(詳細は後述の「工夫」節)。

この限界が**EGM(Endogenous Grid Method, Carroll 2006)**で家計ブロックを丸ごと置き換えると
解消するかどうかを、別ブランチ [`egm-household`](https://github.com/sanrokurintaro/hank-pfi/tree/egm-household)
で検証した(結論:解消する)。そのブランチには`household_egm.py` / `distribution_egm.py` /
`steady_state_egm.py` / `transition_egm.py` / `jacobian_egm.py`と、対応するテスト・実行例が
ある。比較は以下の通り(n_a=150, T=30、自然利子率に-0.02・持続期間4期のショック):

| | `master`(DiscreteDP) | `egm-household`(EGM) |
|---|---|---|
| 家計ソルバー | `household.py`(離散選択PFI) | `household_egm.py`(EGM time iteration) |
| 分布計算 | `distribution.py`(厳密な離散遷移) | `distribution_egm.py`(lottery法、Young 2010) |
| MIT shock遷移 | `transition.py` | `transition_egm.py` |
| Jacobian | `jacobian.py` | `jacobian_egm.py` |
| Jacobian計算に必要な摂動幅 | `dr=0.02, dlogw=0.1`(大きくしないと符号が壊れる) | `dr=dlogw=1e-4`(教科書通りの微小摂動で十分) |
| ノーショック経路のレベルバイアス | n_a=150で2〜4% | 数値誤差程度(事実上ゼロ) |
| Jacobian構築時間 | 42.6秒 | 2.5秒 |
| ZLB遷移の収束 | **収束せず**(`converged=False`、区間探索が尽きてfallbackに落ちた) | 収束(3回の候補で発見) |

ZLBの行列ソルバー自体(`hank_zlb.py`の`solve_hank_transition_path`)はどちらの家計ブロックが
作ったJacobianかを一切問わない純粋な線形代数なので、変更していない。

| | DiscreteDP | EGM |
|---|---|---|
| Jacobian構築時間 | 42.6秒 | 2.5秒 |
| ZLB遷移の収束 | **収束せず**(`converged=False`、区間探索が尽きてfallbackに落ちた) | 収束(3回の候補で発見) |
| J_eff対角成分 | ギザギザ(t=5付近で谷、t=10以降も波打つ) | なめらかに単調減少 |
| output gap・政策金利の経路 | 非単調に大きく振動し、t=15近辺で急落(収束していない解なので信頼できない) | なめらかに定常状態へ収束 |

`dr=0.02`というDiscreteDP側の「大きい摂動幅」は、前回のn_a=150・T=30という設定では、もはや
ノイズを均すのに十分ではなかったことがこの比較で分かる(グリッドサイズ・ホライズンの組み合わせ
次第で必要な摂動幅が変わってしまうこと自体が、この方式の実務上の脆さを物語っている)。EGM側は
教科書通りの微小摂動(`dr=1e-4`)のまま、ホライズンや摂動幅を調整する必要がなかった。

## 階層図
```text
hank-pfi/
├── LICENSE
├── README.md
├── pyproject.toml
├── poetry.toml
├── poetry.lock
├── .gitignore
├── src/
│   └── hank_pfi/
│       ├── __init__.py
│       ├── grids.py           -- ① グリッド・所得過程の離散化
│       ├── household.py       -- ② 家計ソルバー(PFI / DiscreteDP)
│       ├── distribution.py    -- ③ 定常分布 Φ(a,e)
│       ├── aggregate.py       -- ④ 集計 K, C, L
│       ├── steady_state.py    -- ⑤ Aiyagari定常均衡
│       ├── nk_zlb.py          -- ⑥ NK+ZLB(代表的家計)
│       ├── transition.py      -- ⑦ MIT shock遷移
│       ├── jacobian.py        -- ⑧ Sequence-Space Jacobian
│       └── hank_zlb.py        -- ⑨ HANK化ZLB遷移
├── tests/
│   ├── test_household.py
│   ├── test_distribution.py
│   ├── test_steady_state.py
│   ├── test_nk_zlb.py
│   ├── test_transition.py
│   ├── test_jacobian.py
│   └── test_hank_zlb.py
└── examples/
    ├── run_aiyagari.py
    ├── run_zlb_transition.py
    ├── run_hank_zlb_transition.py
    ├── aiyagari_steady_state.png
    ├── zlb_transition.png
    └── hank_vs_rank_zlb.png
```

## アーキテクチャ(層構造)

| 層 | ファイル | 役割 | 使っている既存実装 |
|---|---|---|---|
| ① グリッド・離散化 | [`grids.py`](src/hank_pfi/grids.py) | 資産グリッド生成、所得過程のRouwenhorst離散化 | `quantecon.markov.rouwenhorst` |
| ② 家計ソルバー(PFI) | [`household.py`](src/hank_pfi/household.py) | 状態空間をSA-pair形式のスパース行列で構築し`DiscreteDP`に渡す | `quantecon.markov.DiscreteDP`(`method='policy_iteration'`) |
| ③ 分布 | [`distribution.py`](src/hank_pfi/distribution.py) | 誘導されたMarkov連鎖の定常分布を計算 | `DiscreteDP`が返す`res.mc`を使った独自のpower iteration(理由は後述) |
| ④ 集計 | [`aggregate.py`](src/hank_pfi/aggregate.py) | 分布×政策関数からK, C, Lを集計 | 自作(軽量) |
| ⑤ 定常均衡(Aiyagari GE) | [`steady_state.py`](src/hank_pfi/steady_state.py) | Cobb-Douglas企業のFOCと資本市場清算条件からr*を二分探索 | `scipy.optimize.brentq` |
| ⑥ NKブロック+ZLB遷移(代表的家計版) | [`nk_zlb.py`](src/hank_pfi/nk_zlb.py) | 3本式NKブロック(簡約形)+OccBin型regime-switchingでZLB遷移パスを解く | 発想のみ借用(`occbin`は未使用、独自実装) |
| ⑦ MIT shock遷移 | [`transition.py`](src/hank_pfi/transition.py) | 時間変化する価格経路{r_t,w_t}の下で家計のBellman問題を後ろ向きに1回だけ解く(Howard改善ではなく定常でない有限ホライズン逐次解) | ②の`build_reward_transition`を再利用 |
| ⑧ 家計のSequence-Space Jacobian | [`jacobian.py`](src/hank_pfi/jacobian.py) | ⑦を使い、d log C_t/d r_s と d log C_t/d log w_s のT×T行列を数値微分で構築(Auclert et al. 2021のbrute-force版) | 自作 |
| ⑨ HANK化されたZLB遷移 | [`hank_zlb.py`](src/hank_pfi/hank_zlb.py) | ⑥の代表的家計IS曲線を⑧のJacobianに置き換え、w_t=Y_tの閉じ方でr-チャネルと賃金チャネルを合成、行列版OccBinで解く | 自作(RANK特殊ケースで⑥と厳密一致することをテストで検証) |

## なぜ`MarkovChain.stationary_distributions`を使わなかったか

離散グリッド上の家計問題は「純粋な離散選択」(内挿なし)なので、最適政策が複数の(a,e)状態を
同じ少数のグリッド点に写してしまい、誘導されたMarkov連鎖が**見かけ上reducible**になりやすい
(経済学的な縮退ではなく、グリッドが粗いことによる数値的アーティファクト)。
固有ベクトル解法(`stationary_distributions`)はこの場合どの閉じたクラスを返すか不定になるため、
一様分布からのpower iteration(= Young (2010) の non-stochastic simulationと同じ発想)で
「集団全体が長期的にどこに落ち着くか」を直接計算する方式にした。`distribution.py`は
資産グリッド上限に質量が溜まりすぎていないか(= `a_max`が効きすぎていないか)も自動で警告する。

## セットアップ

[Poetry](https://python-poetry.org/) で依存関係と仮想環境を管理している
([pyproject.toml](pyproject.toml))。

```bash
pipx install poetry   # 未導入の場合
poetry install
poetry run pytest tests/ -v
```

## 実行例

```bash
poetry run python examples/run_aiyagari.py            # 定常均衡(層①〜⑤)
poetry run python examples/run_zlb_transition.py      # ZLB遷移パス(代表的家計版、層⑥)
poetry run python examples/run_hank_zlb_transition.py # ZLB遷移パス(HANK版、層⑦〜⑨)
```

`run_aiyagari.py`は資産分布と貯蓄政策関数の図を、`run_zlb_transition.py`は自然利子率ショックに
対するoutput gap・インフレ率・政策金利の動きを`examples/`以下にPNGで出力する。
`run_hank_zlb_transition.py`は同じ比較をHANK版(家計のJacobianを使ったIS曲線)と代表的家計版で
並べて出力する(家計のSequence-Space Jacobian構築にそれなりに時間がかかる、数十秒程度)。

## 現状できていること

- 家計のBellman問題をPFI(Howard改善)で解き、Aiyagari定常均衡(r\*, w\*, 資産分布)を求める
  (テスト: `tests/test_household.py`, `test_distribution.py`, `test_steady_state.py`)
- ZLBが一時的にのみbindする合理的なimpulse responseを、OccBinと同じregime-switchingロジックで
  再現する(代表的家計版、テスト: `tests/test_nk_zlb.py`)。ノーショック経路が定常状態と一致すること、
  ZLBが解除された後にi_t ≥ 0が常に満たされることを確認済み。
- **家計ブロックから真のHANK IS曲線を構築し、ZLB遷移に接続した**(`jacobian.py`, `transition.py`,
  `hank_zlb.py`)。代表的家計のIS曲線(`sigma`一定)を、②の家計PFIソルバーが生成する
  **Sequence-Space Jacobian**(d log C_t/d r_s, d log C_t/d log w_s)で置き換え、
  `w_t=Y_t`の標準的な閉じ方でr-チャネルと賃金チャネルを合成、ZLBは同じOccBin型の考え方
  (ただし行列版、`_solve_given_regime`)で解く。`hank_zlb.py`の行列ソルバーは、代表的家計と
  数学的に同値な合成Jacobianを入力した場合に`nk_zlb.py`の結果と浮動小数点精度で完全一致することを
  テストで検証済み(`tests/test_hank_zlb.py::test_matches_the_representative_agent_solver_for_an_equivalent_jacobian`)。

## 工夫

HANK化の過程で3つの非自明な問題にぶつかり、それぞれ対処した。

1. **タイミング規約のねじれ**: `household.py`では`r_path[t]`を「tに保有している資産が生む
   当期の収益」として扱っている。このため`jacobian.J_r[t,s]`は、対角成分(s=t)が**正**(当期の
   利子収入が増えるので当期消費も増える、純粋な所得効果)、1つ上の対角成分(s=t+1)が**負**
   (t→t+1の貯蓄の収益率が上がるので当期消費を減らして貯蓄を増やす、標準的な代替効果)という
   構造になる。教科書のNK IS曲線の`r_t`は後者(t→t+1の貯蓄に関連する収益率)に対応するため、
   `hank_zlb.py`では`J_r`を1期分シフトして使っている(`_shift_jacobian`)。検証の途中でこれに
   気づかず対角成分の符号を見て「バグだ」と誤解しかけた経緯がある。
2. **離散グリッドJacobianの数値アーティファクト**: `DiscreteDP`は補間なしの純粋な離散選択なので、
   価格を微小変化させても最適な資産グリッド点が変わらない限り消費は全く反応しない「階段関数」になる。
   逆に価格変化がグリッド1点分の選択フリップを引き起こすと、グリッド間隔分の「ジャンプ」が起きて
   実際の限界的な反応よりはるかに大きなノイズになる(実際、賃金Jacobianの対角成分の符号が
   反転するほど深刻だった)。`transition.py`の`_quadratic_refine_a`で、離散argmax周辺3点に
   放物線を当てはめて連続的な"shadow choice"を復元することで緩和しているが完全ではなく、
   グリッドが粗いほど定常状態からのレベルバイアス(n_a=150で2〜4%程度、n_a=300で1%未満)が残る。
   本番のSequence-Space Jacobianツール(`sequence-jacobian`等)がEGMで補間済みの政策関数を
   前提にしているのは、まさにこの問題を避けるためだと実装してみて初めて実感した。
3. **キャリブレーションのミスマッチと有限ホライズンの境界効果**: 家計ブロックから出てくる
   IS曲線の自己弾力性(`J_eff`の対角、おおよそ-0.03〜-0.1)は、代表的家計の`NKParams.sigma=1.0`
   よりずっと小さい。そのため`nk_zlb.py`のデモと同じ大きさのショック(-0.02)をHANK版に入れても、
   このグリッド・ショック設定の組み合わせではZLBにほぼ到達しない(`examples/run_hank_zlb_transition.py`
   の出力を参照、具体的な数値はグリッドサイズに応じて変動する)。また、残存するJacobianのノイズが
   `(I - M)`の固有値をやや不安定にし、有限ホライズンTの終端付近でoutput gapが跳ね上がる境界効果が
   出ることも確認した(同スクリプトでは図の表示範囲をホライズン前半に絞ることで対処している)。
   これらは「異質な家計モデルは代表的家計モデルよりIS曲線の傾きが緩やかになりうる」という
   HANK文献でよく議論される論点と、離散グリッドJacobianの実務上の脆さの両方を具体的な数値で
   確認できた、という意味で有益な副産物ではある。

## 次の拡張(未実装・既知の制約)

1. **GreenQEの導入**: QEは異質な家計や企業に異なる効果をもたらしうる。
   なので、それをこのモデルを軸に表現したい。
