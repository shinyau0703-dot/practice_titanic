🔍 Titanic EDA 流程
1. 初步檢查

目標：確定資料能讀、欄位有哪些

看 train.csv 的前幾列、欄位名稱

確認資料量（列數、欄位數）

👉 問題：

共有幾筆乘客資料？

欄位有哪一些？

2. 資料結構 & 缺值檢查

目標：知道哪些欄位有缺值、缺多少

用 .info() 看資料型態（int、float、object）

用 .isna().sum() 數缺值

👉 問題：

Age 有多少缺值？

Cabin 幾乎都是缺值？要不要捨棄？

3. 單變數分佈 (Univariate Analysis)

目標：看每個欄位自己長什麼樣子

數值型：年齡直方圖 (Age)、票價直方圖 (Fare)

類別型：統計人數（Sex、Pclass、Embarked）

👉 問題：

男/女比例大概是多少？

年齡分佈偏年輕還是年長？

票價範圍大嗎？有極端值嗎？

4. 雙變數關聯 (Bivariate Analysis)

目標：看看「生存率」與其他欄位的關係

類別 vs 生存率：

Sex vs Survived

Pclass vs Survived

數值 vs 生存率：

Age vs Survived（分箱後比較，例如 <18、18-40、40+）

Fare vs Survived

👉 問題：

女性生存率是不是明顯高？

頭等艙 (Pclass=1) 的生存率是否 > 三等艙？

5. 多變數交互作用 (Optional, 可簡單做)

目標：看看兩個欄位聯合起來，生存有什麼趨勢

Sex + Pclass → 不同性別在不同艙等的生存率

SibSp + Parch → 家庭人數對生存率的影響

👉 問題：

女生 + 頭等艙是不是幾乎都活下來？

家人太多或太少的人，生存率比較低？

6. 總結

目標：整理觀察結果，當作後續特徵工程的依據

哪些欄位很有用（Sex, Pclass, Age, Fare）

哪些欄位缺值太多、要捨棄或簡化（Cabin, Ticket）

哪些欄位需要轉換（類別 → 數值、年齡分箱等）

✅ 小提示

你可以用 pandas + matplotlib/seaborn 畫圖

Notebook (.ipynb) 很適合做 EDA，因為可以一格一格試不同欄位

觀察重點不在「畫圖多漂亮」，而是在於「發現問題 & 洞察」