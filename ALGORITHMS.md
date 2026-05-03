# Kiến thức về các thuật toán sử dụng trong hệ thống

## Tổng quan

Trong đề tài dự đoán tiểu đường, bài toán được xây dựng dưới dạng **phân loại nhị phân** với biến mục tiêu:

| Giá trị | Ý nghĩa |
|---:|---|
| `0` | Không tiểu đường |
| `1` | Có tiểu đường |

Hệ thống sử dụng ba thuật toán học máy có giám sát:

| Thuật toán | Nhóm mô hình | Vai trò |
|---|---|---|
| Logistic Regression | Mô hình tuyến tính | Mô hình nền, dễ giải thích |
| Random Forest | Ensemble Bagging | Học quan hệ phi tuyến, giảm overfitting |
| XGBoost | Ensemble Boosting | Tối ưu hiệu năng dự đoán |

## 1. Logistic Regression

### 1.1. Giới thiệu

**Logistic Regression** là thuật toán học máy có giám sát, thường được sử dụng cho bài toán phân loại nhị phân. Trong đề tài này, Logistic Regression được dùng để dự đoán xác suất một người thuộc nhóm **có tiểu đường** dựa trên các đặc trưng đầu vào như chỉ số BMI, huyết áp cao, cholesterol cao, tuổi, giới tính, tình trạng sức khỏe và hành vi lối sống.

Mục tiêu của mô hình là ước lượng:

```text
P(Diabetes_binary = 1 | X)
```

Trong đó:

- `X` là tập đặc trưng đầu vào.
- `Diabetes_binary = 1` là lớp có tiểu đường.

### 1.2. Ý tưởng hoạt động

Logistic Regression trước hết tính một giá trị tuyến tính từ các đặc trưng đầu vào.

Ở dạng đơn giản với một đặc trưng:

```text
z = mx + b
```

Với nhiều đặc trưng:

```text
z = w1x1 + w2x2 + ... + wnxn + b
```

Trong đó:

- `x1, x2, ..., xn` là các đặc trưng đầu vào.
- `w1, w2, ..., wn` là trọng số mô hình học được.
- `b` là hệ số chặn.
- `z` là giá trị tuyến tính trước khi chuyển thành xác suất.

Sau đó, mô hình đưa `z` qua hàm sigmoid:

```text
p = 1 / (1 + e^(-z))
```

Giá trị `p` nằm trong khoảng từ `0` đến `1`, thể hiện xác suất mẫu thuộc lớp `1`.

Nếu:

```text
p >= 0.5
```

mô hình dự đoán là **có tiểu đường**. Ngược lại, nếu:

```text
p < 0.5
```

mô hình dự đoán là **không tiểu đường**.

### 1.3. Hàm mất mát

Logistic Regression sử dụng hàm mất mát **Log Loss**, còn gọi là **Binary Cross-Entropy**:

```text
Loss = -1/n * Σ [ y log(p) + (1 - y) log(1 - p) ]
```

Trong đó:

- `n` là số mẫu dữ liệu.
- `y` là nhãn thật, nhận giá trị `0` hoặc `1`.
- `p` là xác suất mô hình dự đoán mẫu thuộc lớp `1`.

Hàm mất mát này phạt nặng khi mô hình dự đoán sai với xác suất cao. Trong project, công thức này không được tự viết thủ công mà được sử dụng gián tiếp thông qua lớp `LogisticRegression` của thư viện `scikit-learn`.

### 1.4. Áp dụng trong project

Trong hệ thống, Logistic Regression được cấu hình trong file `src/models/train.py`:

```python
LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    solver="lbfgs",
    random_state=42,
)
```

Ý nghĩa các tham số:

- `class_weight="balanced"`: xử lý mất cân bằng lớp giữa nhóm không tiểu đường và có tiểu đường.
- `max_iter=1000`: số vòng lặp tối đa trong quá trình tối ưu.
- `solver="lbfgs"`: thuật toán tối ưu dùng để tìm bộ trọng số phù hợp.
- `random_state=42`: giúp kết quả có thể tái lập.

Trong quá trình dự đoán, xác suất lớp có tiểu đường được lấy bằng:

```python
model.predict_proba(X_input)[:, 1]
```

Dòng này tương ứng với:

```text
P(Diabetes_binary = 1 | X)
```

### 1.5. Ưu điểm

- Đơn giản, dễ hiểu và dễ triển khai.
- Phù hợp với bài toán phân loại nhị phân.
- Huấn luyện nhanh.
- Dễ giải thích hơn các mô hình phức tạp.
- Phù hợp làm mô hình baseline để so sánh với các thuật toán khác.

### 1.6. Nhược điểm

- Chủ yếu học tốt các quan hệ tuyến tính.
- Khó mô hình hóa các quan hệ phi tuyến phức tạp.
- Có thể cho hiệu năng thấp hơn các mô hình ensemble.
- Nhạy cảm với thang đo dữ liệu, nên cần chuẩn hóa các biến số.

### 1.7. Vai trò trong đề tài

Logistic Regression được sử dụng làm mô hình nền vì bài toán dự đoán tiểu đường là bài toán phân loại nhị phân. Mô hình này giúp cung cấp kết quả tham chiếu đơn giản, dễ giải thích, từ đó so sánh với Random Forest và XGBoost.

## 2. Random Forest

### 2.1. Giới thiệu

**Random Forest** là thuật toán học máy có giám sát thuộc nhóm **ensemble learning**. Thuật toán này kết hợp nhiều cây quyết định để đưa ra kết quả dự đoán cuối cùng. Thay vì chỉ dựa vào một cây quyết định, Random Forest xây dựng một tập hợp nhiều cây, sau đó tổng hợp kết quả của các cây này.

Trong bài toán dự đoán tiểu đường, Random Forest được sử dụng để khai thác các quan hệ phi tuyến giữa các đặc trưng sức khỏe, hành vi lối sống và nhân khẩu học.

### 2.2. Ý tưởng hoạt động

Một cây quyết định đơn lẻ có thể dễ bị overfitting, tức là học quá kỹ dữ liệu huấn luyện và dự đoán kém trên dữ liệu mới. Random Forest giảm hiện tượng này bằng cách tạo ra nhiều cây khác nhau.

Quy trình hoạt động:

```text
1. Tạo nhiều tập dữ liệu con từ tập train bằng bootstrap sampling.
2. Huấn luyện một Decision Tree trên mỗi tập dữ liệu con.
3. Ở mỗi nút chia, chỉ xét một tập con ngẫu nhiên các đặc trưng.
4. Khi dự đoán, mỗi cây đưa ra một nhãn.
5. Kết quả cuối cùng được quyết định bằng bỏ phiếu đa số.
```

Với bài toán phân loại nhị phân:

- Nếu đa số cây dự đoán `1`, mô hình kết luận là có tiểu đường.
- Nếu đa số cây dự đoán `0`, mô hình kết luận là không tiểu đường.

### 2.3. Bagging trong Random Forest

Random Forest sử dụng kỹ thuật **Bagging**, viết tắt của **Bootstrap Aggregating**.

- **Bootstrap**: lấy mẫu ngẫu nhiên có hoàn lại từ tập huấn luyện.
- **Aggregating**: tổng hợp kết quả từ nhiều mô hình con.

Nhờ cơ chế này, Random Forest thường ổn định hơn một cây quyết định đơn lẻ và ít bị ảnh hưởng bởi nhiễu trong dữ liệu.

### 2.4. Áp dụng trong project

Trong hệ thống, Random Forest được cấu hình trong file `src/models/train.py`:

```python
RandomForestClassifier(
    n_estimators=rf_estimators,
    max_depth=16,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=1,
)
```

Ý nghĩa các tham số:

- `n_estimators`: số lượng cây quyết định trong rừng.
- `max_depth=16`: giới hạn độ sâu tối đa của mỗi cây, giúp giảm overfitting.
- `min_samples_leaf=2`: số mẫu tối thiểu tại một nút lá.
- `class_weight="balanced"`: xử lý mất cân bằng lớp.
- `random_state=42`: giúp kết quả có thể tái lập.
- `n_jobs=1`: số tiến trình dùng khi huấn luyện.

### 2.5. Ưu điểm

- Có khả năng học các quan hệ phi tuyến.
- Giảm overfitting so với một cây quyết định đơn lẻ.
- Hoạt động tốt với dữ liệu dạng bảng.
- Ít nhạy cảm với thang đo dữ liệu hơn Logistic Regression.
- Có thể tính độ quan trọng của các đặc trưng.
- Phù hợp với dữ liệu gồm nhiều loại biến như binary, ordinal và numeric.

### 2.6. Nhược điểm

- Khó giải thích hơn Logistic Regression.
- Mô hình lớn hơn và tốn bộ nhớ hơn.
- Dự đoán có thể chậm hơn nếu số lượng cây lớn.
- Không tối ưu lỗi tuần tự như các thuật toán boosting.

### 2.7. Vai trò trong đề tài

Random Forest được sử dụng để so sánh với Logistic Regression và đánh giá khả năng học các mối quan hệ phức tạp hơn. Trong bài toán tiểu đường, nguy cơ mắc bệnh có thể phụ thuộc vào sự kết hợp giữa nhiều yếu tố như BMI, tuổi, huyết áp cao, cholesterol cao, vận động thể chất và lối sống. Random Forest phù hợp để khai thác các tương tác này.

## 3. XGBoost

### 3.1. Giới thiệu

**XGBoost**, viết tắt của **Extreme Gradient Boosting**, là một thuật toán học máy mạnh thuộc nhóm **boosting ensemble**. XGBoost xây dựng nhiều cây quyết định theo cách tuần tự, trong đó mỗi cây mới được huấn luyện để sửa lỗi của các cây trước đó.

Trong các bài toán dữ liệu dạng bảng, XGBoost thường cho hiệu năng cao và được sử dụng rộng rãi trong các bài toán phân loại.

### 3.2. Ý tưởng hoạt động

Khác với Random Forest, các cây trong XGBoost không được huấn luyện độc lập. Thay vào đó, mô hình được xây dựng từng bước.

Quy trình tổng quát:

```text
1. Bắt đầu với một mô hình dự đoán ban đầu.
2. Tính lỗi giữa dự đoán và nhãn thật.
3. Huấn luyện cây tiếp theo để sửa phần lỗi còn lại.
4. Kết hợp cây mới vào mô hình hiện tại.
5. Lặp lại quá trình cho đến khi đạt số cây hoặc điều kiện dừng.
```

Mỗi cây mới tập trung vào những mẫu mà mô hình trước đó dự đoán chưa tốt. Nhờ vậy, XGBoost có khả năng tạo ra mô hình mạnh từ nhiều cây yếu.

### 3.3. Gradient Boosting

XGBoost dựa trên ý tưởng **Gradient Boosting**. Thuật toán tối ưu mô hình bằng cách giảm dần hàm mất mát qua từng vòng lặp. Mỗi cây mới được xây dựng theo hướng làm giảm lỗi dự đoán của mô hình hiện tại.

Với bài toán phân loại nhị phân, XGBoost thường tối ưu hàm mất mát dạng log loss cho xác suất dự đoán.

### 3.4. Regularization

Một điểm mạnh của XGBoost là có cơ chế **regularization** để hạn chế overfitting. Regularization giúp mô hình không trở nên quá phức tạp và cải thiện khả năng tổng quát hóa trên dữ liệu mới.

Ngoài ra, XGBoost còn hỗ trợ các kỹ thuật như:

- Giới hạn độ sâu của cây.
- Điều chỉnh tốc độ học bằng `learning_rate`.
- Lấy mẫu dòng bằng `subsample`.
- Lấy mẫu cột bằng `colsample_bytree`.
- Xử lý mất cân bằng lớp bằng `scale_pos_weight`.

### 3.5. Áp dụng trong project

Trong hệ thống, XGBoost được cấu hình trong file `src/models/train.py`:

```python
XGBClassifier(
    n_estimators=xgb_estimators,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    objective="binary:logistic",
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=1,
)
```

Ý nghĩa các tham số:

- `n_estimators`: số lượng cây boosting.
- `max_depth=4`: giới hạn độ sâu của cây.
- `learning_rate=0.05`: tốc độ học, giúp mô hình cập nhật từ từ và ổn định hơn.
- `subsample=0.9`: tỷ lệ mẫu dòng được sử dụng cho mỗi cây.
- `colsample_bytree=0.9`: tỷ lệ đặc trưng được sử dụng cho mỗi cây.
- `objective="binary:logistic"`: dùng cho bài toán phân loại nhị phân và trả về xác suất.
- `eval_metric="logloss"`: sử dụng log loss làm chỉ số đánh giá trong quá trình huấn luyện.
- `scale_pos_weight`: xử lý mất cân bằng giữa lớp không tiểu đường và có tiểu đường.
- `random_state=42`: giúp kết quả có thể tái lập.

### 3.6. Ưu điểm

- Hiệu năng cao trên dữ liệu dạng bảng.
- Có khả năng học các quan hệ phi tuyến phức tạp.
- Có regularization giúp giảm overfitting.
- Hỗ trợ xử lý mất cân bằng lớp.
- Có thể đánh giá độ quan trọng của đặc trưng.
- Thường đạt kết quả tốt trong các bài toán phân loại.

### 3.7. Nhược điểm

- Có nhiều tham số cần điều chỉnh.
- Khó giải thích hơn Logistic Regression.
- Có thể overfit nếu cấu hình không phù hợp.
- Huấn luyện có thể tốn thời gian hơn Logistic Regression.

### 3.8. Vai trò trong đề tài

XGBoost được sử dụng như một mô hình mạnh để tối ưu hiệu năng dự đoán. Với dữ liệu y tế dạng bảng, XGBoost có khả năng học các quan hệ phức tạp giữa các yếu tố như BMI, huyết áp, cholesterol, tuổi, lối sống và tình trạng sức khỏe tổng quát.

## 4. So sánh ba thuật toán

| Tiêu chí | Logistic Regression | Random Forest | XGBoost |
|---|---|---|---|
| Loại mô hình | Tuyến tính | Ensemble Bagging | Ensemble Boosting |
| Cách học | Học trọng số tuyến tính | Nhiều cây độc lập bỏ phiếu | Nhiều cây tuần tự sửa lỗi |
| Khả năng học phi tuyến | Thấp | Tốt | Rất tốt |
| Khả năng giải thích | Cao | Trung bình | Trung bình đến thấp |
| Tốc độ huấn luyện | Nhanh | Trung bình | Trung bình đến chậm |
| Xử lý mất cân bằng | `class_weight` | `class_weight` | `scale_pos_weight` |
| Vai trò | Baseline | Mô hình phi tuyến ổn định | Mô hình tối ưu hiệu năng |

## 5. Lý do lựa chọn ba thuật toán

Ba thuật toán được lựa chọn nhằm so sánh các hướng tiếp cận khác nhau cho bài toán dự đoán tiểu đường:

- **Logistic Regression** được dùng làm mô hình nền vì đơn giản, dễ giải thích và phù hợp với phân loại nhị phân.
- **Random Forest** được dùng để khai thác các quan hệ phi tuyến và giảm overfitting thông qua nhiều cây quyết định.
- **XGBoost** được dùng để tối ưu hiệu năng dự đoán nhờ cơ chế boosting và regularization.

Việc so sánh ba mô hình giúp đánh giá sự đánh đổi giữa độ chính xác, khả năng phát hiện ca tiểu đường, khả năng giải thích và hiệu năng tổng thể của hệ thống.

## 6. Các chỉ số đánh giá liên quan

Sau khi huấn luyện, các mô hình được đánh giá bằng các chỉ số:

| Chỉ số | Ý nghĩa |
|---|---|
| Accuracy | Tỷ lệ dự đoán đúng trên toàn bộ tập test |
| Precision | Trong các mẫu dự đoán là tiểu đường, tỷ lệ dự đoán đúng |
| Recall | Trong các mẫu thực sự tiểu đường, tỷ lệ được phát hiện đúng |
| F1-score | Trung bình điều hòa giữa Precision và Recall |
| ROC-AUC | Khả năng phân biệt giữa hai lớp ở nhiều ngưỡng khác nhau |
| Average Precision | Chất lượng mô hình trên đường Precision-Recall |

Vì dataset bị mất cân bằng lớp, không nên chỉ dựa vào Accuracy. Trong bài toán y tế, **Recall**, **F1-score** và **ROC-AUC** cần được quan tâm nhiều hơn để hạn chế bỏ sót các trường hợp có nguy cơ tiểu đường.

