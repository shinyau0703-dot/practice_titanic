```mermaid
flowchart TB
  %% 結構（放大版）
  subgraph DB["titanic 資料庫（schema: public）"]
    R["public.runs（核心成果表）"]
    F["public.fold_results（每折結果表）"]
    T["public.train_clean（匯入的訓練資料）"]
  end

  subgraph ROLES["角色（誰在操作）"]
    P["postgres（超級使用者 / 擁有者）"]
    A["app / practice_titanic（一般帳號）"]
    G["PUBLIC（所有人）"]
  end

  %% 擁有關係
  P -->|建立 / Owner| R
  P -->|建立 / Owner| T
  P -.可能.-> F
  A -.可能.-> F

  class P high; class R,T medium; class F,A low; class G none;
  classDef high   fill:#ffe0e0,stroke:#d33,stroke-width:1.6px,color:#600;
  classDef medium fill:#fff4cc,stroke:#d9a000,stroke-width:1.2px,color:#664400;
  classDef low    fill:#e7f7e7,stroke:#2d8a34,stroke-width:1.2px,color:#184d24;
  classDef none   fill:#f2f2f2,stroke:#aaa,color:#666;


```
--- 

postgres = 系統管理員，管全部。

app/practice_titanic = 專案成員，用來日常跑 SQL、管理專案表。

PUBLIC = 廣義的所有人，最好收掉建表權限，避免亂象。

--- 

postgres 是超管，擁有所有東西。
### 1. postgres（超級使用者）

就像房東＋管理員合一，

這個帳號天生什麼都能做：建資料庫、建資料櫃子（schema）、建資料表、改密碼、刪資料……全部權限都在它手上。

一般只拿來做「管理」或「設定」，不會每天拿來跑專案。

### 2. app / practice_titanic（專案用帳號）

想像成「專案成員」：房東（postgres）給這個人一把鑰匙，可以打開 public 這個櫃子，裡面有專案的表格。

權限設計成「夠用就好」：

可以打開櫃子 → 查得到表。

可以讀、寫 → 在三張表 (runs / fold_results / train_clean) 裡新增、修改、查詢資料。

如果房東願意，還可以讓他自己在櫃子裡多擺一張新表。

好處：日常分析、匯入匯出資料，全靠這個帳號，安全又不會動到系統設定。

### 3. PUBLIC（所有人）

PostgreSQL 預設會給「所有人」一個權限：大家都能在 public 櫃子裡建表。

這樣做風險很大 → 任何新帳號進來都能隨便加東西，像是公共廚房誰都能塞東西一樣。

所以實務上一定會「收回」這個能力，只讓特定專案帳號（例如 app / practice_titanic）可以動。

換句話說：PUBLIC 最好只保留「能看」的權限，不要「能建」的權限。

--- 


```mermaid

flowchart TB
  %% 容器層
  subgraph C["Docker 容器：pg_main"]
    subgraph S["PostgreSQL 16 服務"]
      %% 資料庫
      subgraph DB["資料庫：titanic"]
        %% public（公共區域）
        subgraph PUBLIC["Schema：public（公共區域）"]
          EXT["extensions / 函式（例如 uuid-ossp 等）"]
          UTIL["共用小工具/參照表（可選）"]
        end

        %% app（專案區域）
        subgraph APP["Schema：app（專案區域｜專案資料都放這）"]
          RUNS["runs（每次訓練/實驗的彙總結果）"]
          FOLDS["fold_results（每折指標，FK → runs.id）"]
          TRAIN["train_clean（清理後的訓練資料）"]
          TEST["test（測試資料）"]
          SUB["submission（預測提交檔）"]
          VIEWS["v_*（檢視/報表用彙總，可選）"]
        end
      end
    end
  end

  %% 簡單連線
  C --> S --> DB
  PUBLIC -. 僅放共用/系統物件 .- DB
  APP -. 專案商業資料集中放這裡 .- DB

  %% 樣式
  classDef box fill:#fff,stroke:#999,stroke-width:1.2px,color:#333;
  classDef pub fill:#f2f2f2,stroke:#9e9e9e,color:#555;
  classDef app fill:#e7f7e7,stroke:#2d8a34,color:#184d24;
  class C,S,DB box
  class PUBLIC pub
  class APP,RUNS,FOLDS,TRAIN,TEST,SUB,VIEWS app

```



```mermaid
flowchart TB
  P["postgres（超級使用者）"]
  A["app / practice_titanic（一般帳號）"]
  R["public.runs（成果表）"]
  F["public.fold_results（每折結果表）"]
  T["public.train_clean（清理後的訓練資料）"]
  PUB["PUBLIC（所有人）"]
  SCH["schema public（資料櫃子）"]

  %% 給一般帳號的權限
  P -->|允許 A 可以打開 public 櫃子來用| SCH
  P -->|（可選）允許 A 可以在 public 櫃子裡自己建表| SCH
  P -->|允許 A 可以讀、寫 runs 表| R
  P -->|允許 A 可以讀、寫 fold_results 表| F
  P -->|允許 A 可以讀、寫 train_clean 表| T

  %% 收斂 PUBLIC
  PUB -.-x |取消所有人都能隨便在 public 櫃子建表的權限| SCH

  class P high; class R,T medium; class F,A low; class PUB none; class SCH medium;
  classDef high   fill:#ffe0e0,stroke:#d33,color:#600;
  classDef medium fill:#fff4cc,stroke:#d9a000,color:#664400;
  classDef low    fill:#e7f7e7,stroke:#2d8a34,color:#184d24;
  classDef none   fill:#f2f2f2,stroke:#aaa,color:#666;



```


```mermaid
flowchart TB
  R["public.runs (id PK, timestamp, model_family, avg_acc, avg_f1)"]
  F["public.fold_results (id PK, run_id FK→runs.id, fold, acc, f1, ... )"]
  T["public.train_clean (... features ...)"]

  A["app / practice_titanic（一般帳號）"]

  A -->|INSERT / SELECT / UPDATE / DELETE| R
  A -->|INSERT / SELECT / UPDATE / DELETE| F
  A -->|SELECT（通常）或 INSERT（若要清洗後寫回）| T

  F -.->|run_id 參照 runs.id| R

  class R,T medium; class F,A low;
  classDef medium fill:#fff4cc,stroke:#d9a000,color:#664400;
  classDef low    fill:#e7f7e7,stroke:#2d8a34,color:#184d24;



```



```mermaid
flowchart LR
  subgraph S1["public (schema)"]
    A["像一個櫃子"]
    B["表都放在這裡<br/>例: public.runs"]
  end

  subgraph S2["PUBLIC (角色群組)"]
    C["代表所有使用者"]
    D["給它權限 = 全部人都能用"]
  end

  subgraph S3["app (帳號)"]
    E["專屬你的使用者帳號"]
    F["只給專案需要的權限"]
  end

```







