# 🏦 Nova Bank - Mô Hình Dự Đoán Nợ Xấu & Thẩm Định Tín Dụng Tự Động

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Gradio](https://img.shields.io/badge/UI-Gradio-orange.svg)](https://gradio.app/)
[![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-green.svg)](https://scikit-learn.org/)

Dự án ứng dụng Khoa học dữ liệu (Data Science) và Học máy (Machine Learning) toàn diện nhằm nhận diện các "điểm đen" rủi ro tín dụng và triển khai giao diện chấm điểm tín dụng tự động cho các danh mục vay cá nhân, y tế, giáo dục và kinh doanh.

---

## 📌 1. Bài Toán Kinh Doanh & Mục Tiêu

### Bối cảnh
Nova Bank là một tổ chức tài chính hoạt động trên khắp Hoa Kỳ, Vương quốc Anh và Canada, quản lý tệp dữ liệu lớn với hơn 32.000 hồ sơ vay vốn. Trong bối cảnh kinh tế biến động, ngân hàng đứng trước cơ hội mở rộng dư nợ để tăng trưởng nhưng cũng đối mặt với thách thức lớn về việc kiểm soát tỷ lệ nợ xấu (NPL).

### Thách thức kinh doanh
* **Cân bằng giữa Tăng trưởng & Rủi ro:** Làm sao để phê duyệt nhiều hồ sơ hơn nhưng không làm gia tăng tỷ lệ bùng nợ.
* **Hạn chế của phương pháp cũ:** Quy trình chấm điểm truyền thống có thể chậm và bỏ sót các mối quan hệ phi tuyến tính giữa các biến rủi ro.

### Mục tiêu dự án
* **Khám phá dữ liệu (EDA):** Giải mã chân dung chi tiết của nhóm khách hàng vỡ nợ ("Defaulter") và cô lập các tín hiệu cảnh báo đỏ.
* **Tối ưu hóa dữ liệu:** Tìm ra các biến số có "sức mạnh dự báo" cao nhất dựa trên chỉ số Information Value (IV) và Weight of Evidence (WoE).
* **Xây dựng màng lọc thông minh:** Huấn luyện và đánh giá các mô hình học máy để dự đoán xác suất vỡ nợ của hồ sơ mới, tối ưu hóa theo chi phí rủi ro tài chính của ngân hàng.

---

## 📊 2. Quy Mô Dữ Liệu & Quy Trình Xử Lý (Data Pipeline)

Dự án sử dụng tập dữ liệu **Credit Risk Dataset** gồm **32.581 hồ sơ vay vốn** với **29 cột dữ liệu** thuộc các nhóm: Thông tin nhân khẩu học, Tình trạng tài chính, Đặc điểm khoản vay và Lịch sử hành vi tín dụng.

### Quy trình làm sạch & Biến đổi dữ liệu
1.  **Xử lý dữ liệu thiếu (Missing Values):**
    * `person_emp_length` (Số năm kinh nghiệm): Điền bằng giá trị **Trung vị (Median)** theo từng loại hình nghề nghiệp (`employment_type`).
    * `loan_int_rate` (Lãi suất vay): Điền bằng giá trị **Trung vị (Median)** của từng hạng vay (`loan_grade`).
2.  **Xử lý dữ liệu nhập sai (Wrong Data):**
    * Loại bỏ các giá trị bất thường do lỗi nhập liệu (Ví dụ: khách hàng 21-22 tuổi nhưng có 123 năm kinh nghiệm, hoặc khách hàng có số tuổi $> 100$).
3.  **Kỹ nghệ đặc trưng (Feature Engineering):**
    * **Xử lý đa cộng tuyến (Multicollinearity):** Loại bỏ các biến bị phụ thuộc lẫn nhau (`loan_to_income_ratio`, `debt_to_income_ratio`) nhằm đảm bảo tính ổn định cho mô hình.
    * **Mã hóa dữ liệu:** Áp dụng phương pháp mã hóa **Weight of Evidence (WoE)** cho các biến phân loại.
    * **Xử lý mất cân bằng dữ liệu:** Tỷ lệ phân phối gốc bị lệch nghiêm trọng (78.2% nợ tốt vs. 21.8% nợ xấu). Kỹ thuật **SMOTE** được áp dụng giúp cân bằng tập huấn luyện thành công (tăng từ 26 Bak dòng lên 40.746 dòng với tỷ lệ 50:50).

---

## 🔍 3. Tín Hiệu Đỏ & Chân Dung Khách Hàng Vỡ Nợ

Thông qua phân tích EDA và Chỉ số giá trị thông tin (IV), sức mạnh dự báo của các biến được phân loại rõ ràng:

* **Yếu tố quyết định (Very Strong / Strong IV):** Tỷ lệ vay/Thu nhập (`loan_percent_income`), Lãi suất (`loan_int_rate`), Xếp hạng khoản vay nội bộ (`loan_grade`), Thu nhập (`person_income`), và Tình trạng sở hữu nhà (`person_home_ownership`).
* **Yếu tố bổ sung (Medium IV):** Lịch sử từng vỡ nợ (`cb_person_default_on_file`) và Mục đích vay (`loan_intent`).

### 🚨 Chân dung "Defaulter" điển hình:
* **Gánh nặng tài chính lớn:** Rủi ro vỡ nợ tăng vọt theo cấp số nhân ngay khi **Tỷ lệ khoản vay trên thu nhập vượt ngưỡng 30%**.
* **Nhóm nhân khẩu học rủi ro:** Khách hàng đang phải **thuê nhà (`RENT`)** kết hợp với xếp hạng tín dụng nội bộ thấp (**Hạng D đến G**) có xác suất nợ xấu cực kỳ cao.
* **Mục đích vay nhạy cảm:** Tỷ lệ bùng nợ tập trung lớn ở các khoản vay cho mục đích **Y tế (Medical)** và **Đảo nợ (Debt Consolidation)**.
* **Bẫy lãi suất:** Nhóm vỡ nợ thường phải chịu mức lãi suất trung vị cao hơn đáng kể (~13.5%) so với nhóm an toàn (~10.7%).

---

## 🤖 4. Huấn Luyện & Đánh Giá Mô Hình

Dữ liệu được phân tách theo tỷ lệ **80% Training / 20% Testing**. Năm thuật toán đã được cấu hình, tối ưu hóa siêu tham số (Hyperparameter Tuning) và thử nghiệm:

| Mô Hình | Recall (Class 1) | Precision (Class 1) | F1-Score (Class 1) |
| :--- | :---: | :---: | :---: |
| Logistic Regression | 0.57 | **0.79** | 0.66 |
| KNN | 0.59 | 0.77 | 0.67 |
| Support Vector Machine (SVC) | 0.59 | 0.77 | 0.67 |
| Decision Tree | 0.74 | 0.74 | 0.74 |
| **Random Forest (Chọn)** | **0.89** | 0.72 | **0.79** |

### 🎯 Chiến lược lựa chọn mô hình
Trong bài toán rủi ro tín dụng ngân hàng, việc "bỏ sót" một khách hàng nợ xấu gây thiệt hại nặng nề hơn nhiều so với việc "từ chối nhầm" một khách hàng tốt. 

Mô hình **Random Forest** được lựa chọn làm mô hình tối ưu nhất nhờ đạt **F1-Score cao nhất (0.79)** và khả năng bắt nợ xấu **(Recall = 0.89)** vượt trội. Lựa chọn này giúp ngân hàng tối ưu hóa bài toán chi phí:
1.  **Giảm thiểu chi phí bỏ sót ca vỡ nợ:** Tránh mất trắng vốn gốc.
2.  **Tối ưu hóa chi phí cơ hội:** Hạn chế tối đa việc từ chối nhầm các khách hàng tốt, giữ vững doanh thu.

---

## 🚀 5. Đề Xuất Triển Khai Thực Tế (Deployment)

### 🖥️ Ứng dụng thẩm định tự động

### 📈 Đề xuất tích hợp hệ thống Core-Banking:
1.  **Scoring API:** Triển khai mô hình Random Forest dưới dạng một REST API thời gian thực để tích hợp trực tiếp vào hệ thống duyệt vay tự động.
2.  **Hard Rules (Quy tắc chặn cứng):** Thiết lập quy tắc tự động từ chối ngay lập tức đối với nhóm rủi ro cực hạn: Khách hàng **Thuê nhà (`RENT`) + Tỷ lệ vay/Thu nhập > 30% + Lãi suất > 14%**.
3.  **Ngưỡng quyết định linh hoạt (Dynamic Threshold):** Cho phép bộ phận Quản trị rủi ro điều chỉnh ngưỡng xác suất dự báo tùy theo chiến lược thắt chặt hay mở rộng tín dụng của ngân hàng trong từng thời kỳ.

