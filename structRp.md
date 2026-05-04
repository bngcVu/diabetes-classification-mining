CHƯƠNG I: GIỚI THIỆU
  1.1 Data Mining là gì?
  1.2 Tổng quan bài toán
  1.3 Mục tiêu nghiên cứu

CHƯƠNG II: GIỚI THIỆU DỮ LIỆU
  2.1 Nguồn dữ liệu (CDC)
  2.2 Mô tả các thuộc tính
  2.3 Phân phối nhãn (mất cân bằng ~14%)

CHƯƠNG III: TIỀN XỬ LÝ DỮ LIỆU
  3.1 Kiểm tra chất lượng dữ liệu
      3.1.1 Missing Values
      3.1.2 Duplicates
      3.1.3 Outliers
  3.2 Feature Engineering
      3.2.1 BMI_category (Binning)
      3.2.2 comorbidity_score (Aggregation)
      3.2.3 healthy_lifestyle (Aggregation)
  3.3 Feature Transformation
      3.3.1 Scaling (continuous_cols)
      3.3.2 Encoding (categorical_cols)
      3.3.3 Binary (giữ nguyên)
  3.4 Xử lý mất cân bằng lớp
      3.4.1 SMOTE
      3.4.2 Class Weight
  3.5 Train/Test Split
      3.5.1 Random Sampling
      3.5.2 Stratified Sampling ✅

CHƯƠNG IV: BÀI TOÁN PHÂN LOẠI
  4.1 Giới thiệu chung về bài toán phân loại
  4.2 Lý do sử dụng bài toán phân loại
  4.3 Thuật toán sử dụng
      4.3.1 Logistic Regression
      4.3.2 Random Forest
      4.3.3 XGBoost

CHƯƠNG V: MÔ HÌNH VÀ THỰC NGHIỆM
  5.1 Thiết lập thực nghiệm
      5.1.1 Môi trường thực nghiệm
      5.1.2 Tham số các mô hình
  5.2 Các chỉ số đánh giá
      5.2.1 Accuracy
      5.2.2 Precision
      5.2.3 Recall
      5.2.4 F1-score
      5.2.5 ROC-AUC
  5.3 Kết quả so sánh 3 thuật toán
      5.3.1 Bảng so sánh đầy đủ
      5.3.2 Biểu đồ trực quan
  5.4 Lý do chọn XGBoost
      5.4.1 Dựa trên kết quả so sánh
      5.4.2 ROC-AUC cao nhất (0.8266)
      5.4.3 Recall tốt nhất (0.7913)
      5.4.4 Phù hợp bài toán y tế

CHƯƠNG VI: KẾT QUẢ VÀ THẢO LUẬN
  6.1 Kết quả chi tiết XGBoost
      6.1.1 Kết quả trên tập test
      6.1.2 Ma trận nhầm lẫn (Confusion Matrix)
      6.1.3 Đường cong ROC
  6.2 Hệ thống Admin & Retrain
      6.2.1 Trang quản trị Admin
      6.2.2 Luồng Upload CSV
            - Kiểm tra định dạng & cột
            - Kiểm tra trùng lặp
            - Lưu lịch sử upload
      6.2.3 Luồng Retrain
            - Kiểm tra điều kiện
            - Gộp dữ liệu cũ + mới
            - Tiền xử lý
            - Train XGBoost
            - So sánh & Quyết định
            - Lưu lịch sử Retrain
      6.2.4 Kết quả Retrain thực tế
            - Phân tích kết quả so sánh
            - Lý do giữ/cập nhật mô hình
  6.3 Phân tích sai số
  6.4 Đề xuất cải tiến

CHƯƠNG VII: KẾT LUẬN
  7.1 Tổng kết kết quả đạt được
  7.2 Hạn chế của nghiên cứu
  7.3 Hướng phát triển trong tương lai

