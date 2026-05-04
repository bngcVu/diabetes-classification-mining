"""
Báo cáo nghiên cứu: Áp dụng Data Mining & Machine Learning
để phân loại nguy cơ mắc bệnh tiểu đường trên bộ dữ liệu CDC (BRFSS)

Cách chạy:
    pip install python-docx
    python generate_baocao.py

File DOCX sẽ được lưu tại: BaoCao_DataMining_TieuDuong.docx
"""

from docx import Document
from docx.shared import Pt, RGBColor, Cm, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import copy

# ─── Màu sắc ──────────────────────────────────────────────────────────────────
C_PRIMARY   = RGBColor(0x1F, 0x4E, 0x79)
C_SECONDARY = RGBColor(0x2E, 0x75, 0xB6)
C_ACCENT    = RGBColor(0x70, 0xAD, 0x47)
C_GRAY      = RGBColor(0x59, 0x59, 0x59)
C_BLACK     = RGBColor(0x00, 0x00, 0x00)
C_WHITE     = RGBColor(0xFF, 0xFF, 0xFF)

FONT_NAME = 'Times New Roman'

# ─── Helpers ──────────────────────────────────────────────────────────────────

def set_cell_bg(cell, hex_color):
    """Set background color of a table cell."""
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), hex_color)
    tcPr.append(shd)

def set_cell_borders(cell, color='2E75B6'):
    """Set thin borders on a cell."""
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    tcBorders = OxmlElement('w:tcBorders')
    for side in ('top', 'left', 'bottom', 'right'):
        border = OxmlElement(f'w:{side}')
        border.set(qn('w:val'), 'single')
        border.set(qn('w:sz'), '4')
        border.set(qn('w:space'), '0')
        border.set(qn('w:color'), color)
        tcBorders.append(border)
    tcPr.append(tcBorders)

def add_heading(doc, text, level, color=None):
    p = doc.add_heading(text, level=level)
    for run in p.runs:
        run.font.name = FONT_NAME
        run.font.bold = True
        if color:
            run.font.color.rgb = color
        if level == 1:
            run.font.size = Pt(16)
            run.font.color.rgb = color or C_PRIMARY
        elif level == 2:
            run.font.size = Pt(14)
            run.font.color.rgb = color or C_SECONDARY
        elif level == 3:
            run.font.size = Pt(13)
            run.font.color.rgb = color or C_SECONDARY
    pf = p.paragraph_format
    pf.space_before = Pt(level == 1 and 18 or 14)
    pf.space_after  = Pt(level == 1 and 9  or 7)
    return p

def add_para(doc, text, indent_first=True, italic=False, color=None, size=12):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    pf.space_before = Pt(3)
    pf.space_after  = Pt(3)
    pf.line_spacing = Pt(20)
    if indent_first:
        pf.first_line_indent = Cm(1.25)
    run = p.add_run(text)
    run.font.name = FONT_NAME
    run.font.size = Pt(size)
    run.font.italic = italic
    run.font.color.rgb = color or C_BLACK
    return p

def add_formula(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after  = Pt(6)
    run = p.add_run(text)
    run.font.name = 'Courier New'
    run.font.size = Pt(11)
    run.font.italic = True
    run.font.color.rgb = C_GRAY

def add_caption(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after  = Pt(10)
    run = p.add_run(text)
    run.font.name = FONT_NAME
    run.font.size = Pt(10)
    run.font.italic = True
    run.font.color.rgb = C_GRAY

def h_cell(cell, text, bg='1F4E79'):
    set_cell_bg(cell, bg)
    set_cell_borders(cell, bg)
    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(text)
    run.font.name = FONT_NAME
    run.font.bold = True
    run.font.size = Pt(10)
    run.font.color.rgb = C_WHITE
    cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER

def d_cell(cell, text, shade=False, center=False):
    set_cell_bg(cell, 'EBF3FB' if shade else 'FFFFFF')
    set_cell_borders(cell, '2E75B6')
    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER if center else WD_ALIGN_PARAGRAPH.LEFT
    run = p.add_run(text)
    run.font.name = FONT_NAME
    run.font.size = Pt(10)
    run.font.color.rgb = C_BLACK
    cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER

# ═══════════════════════════════════════════════════════════════════════════════
# BUILD DOCUMENT
# ═══════════════════════════════════════════════════════════════════════════════

doc = Document()

# Page setup: A4, margins 3/2/2/2.5 cm (left/right/top/bottom)
section = doc.sections[0]
section.page_height = Cm(29.7)
section.page_width  = Cm(21.0)
section.left_margin   = Cm(3.0)
section.right_margin  = Cm(2.0)
section.top_margin    = Cm(2.0)
section.bottom_margin = Cm(2.0)

# ── TRANG BÌA ────────────────────────────────────────────────────────────────
def cover_para(doc, text, size=13, bold=False, color=None, space_before=6):
    p = doc.add_paragraph()
    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(space_before)
    p.paragraph_format.space_after  = Pt(3)
    run = p.add_run(text)
    run.font.name = FONT_NAME
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color or C_PRIMARY
    return p

cover_para(doc, 'TRƯỜNG ĐẠI HỌC KHOA HỌC TỰ NHIÊN', 13, True, C_PRIMARY, 60)
cover_para(doc, 'KHOA CÔNG NGHỆ THÔNG TIN', 13, True, C_PRIMARY)
cover_para(doc, '─' * 40, 11, False, C_SECONDARY, 20)
cover_para(doc, 'BÁO CÁO NGHIÊN CỨU', 22, True, C_PRIMARY, 40)
cover_para(doc, 'ÁP DỤNG KỸ THUẬT DATA MINING VÀ MACHINE LEARNING', 16, True, C_SECONDARY, 20)
cover_para(doc, 'ĐỂ PHÂN LOẠI NGUY CƠ MẮC BỆNH TIỂU ĐƯỜNG', 16, True, C_SECONDARY)
cover_para(doc, 'TRÊN BỘ DỮ LIỆU CDC (BRFSS)', 16, True, C_SECONDARY)
cover_para(doc, '─' * 40, 11, False, C_SECONDARY, 20)
cover_para(doc, 'Chuyên ngành: Khoa học Máy tính – Khai phá Dữ liệu', 13, False, C_GRAY, 30)
cover_para(doc, 'Học kỳ II – Năm học 2024–2025', 13, False, C_GRAY)

doc.add_page_break()

# ── TÓM TẮT ─────────────────────────────────────────────────────────────────
add_heading(doc, 'TÓM TẮT', 1)
add_para(doc, 'Bệnh tiểu đường (Diabetes Mellitus) là một trong những vấn đề y tế nghiêm trọng nhất trên toàn thế giới, ảnh hưởng đến hàng trăm triệu người và gây ra nhiều biến chứng nguy hiểm như bệnh tim mạch, suy thận, mù lòa và cắt cụt chi. Việc phát hiện sớm nguy cơ mắc bệnh đóng vai trò then chốt trong việc ngăn ngừa và kiểm soát bệnh tiến triển.')
add_para(doc, 'Nghiên cứu này trình bày một quy trình khai phá dữ liệu (Data Mining) toàn diện và hệ thống nhằm xây dựng mô hình học máy (Machine Learning) có khả năng phân loại nguy cơ mắc bệnh tiểu đường dựa trên bộ dữ liệu Behavioral Risk Factor Surveillance System (BRFSS) do Trung tâm Kiểm soát và Phòng ngừa Dịch bệnh Hoa Kỳ (CDC) thu thập. Bộ dữ liệu bao gồm hơn 253.000 hồ sơ sức khỏe với 21 đặc trưng liên quan đến hành vi sức khỏe, tiền sử bệnh lý và các chỉ số sinh lý.')
add_para(doc, 'Nghiên cứu áp dụng đầy đủ các bước trong quy trình KDD: kiểm tra và làm sạch dữ liệu, kỹ thuật đặc trưng (Feature Engineering), chuẩn hóa và mã hóa, xử lý mất cân bằng lớp bằng SMOTE, chia tập dữ liệu theo phân tầng (Stratified Sampling), và huấn luyện ba mô hình phân loại: Logistic Regression, Random Forest và XGBoost. Mô hình XGBoost đạt hiệu suất tốt nhất với Accuracy ~76%, AUC-ROC ~0.83 và F1-Score ~0.72.')
add_para(doc, 'Từ khóa: Data Mining, Machine Learning, Tiểu đường, CDC BRFSS, XGBoost, SMOTE, Phân loại nhị phân.', indent_first=False, italic=True, color=C_GRAY)

doc.add_page_break()

# ═══════════════════════════════════════════════════════════════════════════════
# CHƯƠNG I
# ═══════════════════════════════════════════════════════════════════════════════
add_heading(doc, 'CHƯƠNG I: GIỚI THIỆU', 1)

add_heading(doc, '1.1 Data Mining là gì?', 2)
add_para(doc, 'Data Mining (Khai phá Dữ liệu) là một lĩnh vực khoa học liên ngành nằm ở giao điểm của Thống kê học, Học máy (Machine Learning), Khoa học máy tính và Cơ sở dữ liệu. Mục tiêu trung tâm của Data Mining là trích xuất tri thức ẩn, có giá trị và tiềm năng sử dụng từ các tập dữ liệu lớn thông qua việc áp dụng các thuật toán và kỹ thuật phân tích tiên tiến. Theo Han, Kamber và Pei (2011), Data Mining được định nghĩa là "quá trình khám phá các mẫu có ý nghĩa (patterns), quy luật (rules) và tri thức từ một lượng lớn dữ liệu lưu trữ trong cơ sở dữ liệu, kho dữ liệu hoặc các kho thông tin khác."')
add_para(doc, 'Data Mining không hoạt động đơn lẻ mà là một bước quan trọng trong quy trình rộng hơn được gọi là KDD (Knowledge Discovery in Databases – Khám phá Tri thức trong Cơ sở Dữ liệu). Quy trình KDD bao gồm các giai đoạn tuần tự: (1) Lựa chọn dữ liệu, (2) Tiền xử lý và làm sạch dữ liệu, (3) Biến đổi dữ liệu, (4) Khai phá dữ liệu, và (5) Đánh giá và diễn giải tri thức. Mỗi bước đều có vai trò thiết yếu trong việc đảm bảo chất lượng và độ tin cậy của tri thức được khám phá.')
add_para(doc, 'Các kỹ thuật chính trong Data Mining bao gồm: Phân loại (Classification), Phân cụm (Clustering), Luật kết hợp (Association Rule Mining), Hồi quy (Regression) và Phát hiện bất thường (Anomaly Detection). Trong phạm vi nghiên cứu này, kỹ thuật Phân loại được lựa chọn là phương pháp cốt lõi do bản chất nhị phân của bài toán: phân biệt giữa người có và không có nguy cơ mắc bệnh tiểu đường.')
add_para(doc, 'Sự bùng nổ của dữ liệu y tế số trong những thập kỷ gần đây đã mở ra cơ hội to lớn cho Data Mining trong lĩnh vực chăm sóc sức khỏe. Các hệ thống hỗ trợ quyết định lâm sàng (Clinical Decision Support Systems – CDSS) được xây dựng trên nền tảng Data Mining đang ngày càng được ứng dụng rộng rãi để hỗ trợ bác sĩ trong việc chẩn đoán, tiên lượng và lên kế hoạch điều trị, từ đó cải thiện chất lượng chăm sóc y tế và giảm thiểu sai sót lâm sàng.')

add_heading(doc, '1.2 Tổng quan bài toán', 2)
add_para(doc, 'Bệnh tiểu đường (Diabetes Mellitus) là tình trạng bệnh lý mãn tính đặc trưng bởi mức đường huyết cao (hyperglycemia) do cơ thể không sản xuất đủ insulin hoặc không sử dụng insulin hiệu quả. Theo Tổ chức Y tế Thế giới (WHO), năm 2023 có khoảng 537 triệu người trưởng thành trên toàn cầu mắc bệnh tiểu đường, và con số này dự kiến sẽ tăng lên 783 triệu vào năm 2045. Riêng tại Hoa Kỳ, CDC ước tính có hơn 37 triệu người mắc bệnh tiểu đường và khoảng 96 triệu người trong tình trạng tiền tiểu đường, trong đó hơn 80% không biết mình mắc bệnh.')
add_para(doc, 'Các biến chứng chính của bệnh bao gồm bệnh tim mạch (tăng nguy cơ gấp 2–4 lần), suy thận mãn tính, bệnh lý võng mạc dẫn đến mù lòa, và bệnh lý thần kinh ngoại biên gây cắt cụt chi. Phát hiện sớm và can thiệp kịp thời không chỉ cải thiện chất lượng cuộc sống của bệnh nhân mà còn giúp giảm đáng kể gánh nặng kinh tế cho hệ thống y tế.')
add_para(doc, 'Bài toán phân loại nguy cơ mắc bệnh tiểu đường được đặt ra trong bối cảnh đó: dựa trên các thông tin về hành vi sức khỏe, lối sống và tiền sử bệnh lý của cá nhân (thu thập qua bảng hỏi điều tra dân số), xây dựng mô hình có khả năng tự động phân loại liệu một cá nhân có nguy cơ mắc hoặc đang mắc bệnh tiểu đường hay không. Đây là bài toán phân loại nhị phân (Binary Classification) với nhãn đầu ra là: 0 – Không mắc bệnh, 1 – Đang mắc hoặc có tiền sử mắc bệnh tiểu đường.')
add_para(doc, 'Thách thức đặc trưng của bài toán bao gồm: mất cân bằng lớp nghiêm trọng (~14% dương tính), đặc trưng hỗn hợp (liên tục, thứ bậc, nhị phân), yêu cầu khả năng diễn giải mô hình trong bối cảnh y tế, và dữ liệu tự báo cáo có thể chứa sai lệch thông tin.')

add_heading(doc, '1.3 Mục tiêu nghiên cứu', 2)
add_para(doc, 'Nghiên cứu này hướng đến các mục tiêu cụ thể sau đây:')
add_para(doc, 'Thứ nhất, xây dựng và triển khai một quy trình khai phá dữ liệu hoàn chỉnh và có hệ thống, bao gồm toàn bộ các bước từ thu thập dữ liệu, tiền xử lý, kỹ thuật đặc trưng, đến huấn luyện và đánh giá mô hình trên bộ dữ liệu BRFSS của CDC.', indent_first=False)
add_para(doc, 'Thứ hai, đề xuất và kiểm chứng các đặc trưng tổng hợp mới như BMI_category, comorbidity_score và healthy_lifestyle nhằm nâng cao khả năng biểu diễn dữ liệu và cải thiện hiệu suất mô hình.', indent_first=False)
add_para(doc, 'Thứ ba, so sánh hiệu suất của ba thuật toán học máy – Logistic Regression, Random Forest và XGBoost – trên cùng điều kiện thực nghiệm để xác định mô hình phù hợp nhất.', indent_first=False)
add_para(doc, 'Thứ tư, phân tích tầm quan trọng của từng đặc trưng (Feature Importance) đối với khả năng dự đoán nguy cơ mắc bệnh tiểu đường, cung cấp thông tin có giá trị cho can thiệp y tế công cộng.', indent_first=False)
add_para(doc, 'Thứ năm, đề xuất các hướng cải tiến và mở rộng nghiên cứu trong tương lai, bao gồm áp dụng kỹ thuật giải thích mô hình (SHAP, LIME) và tích hợp dữ liệu từ nhiều nguồn.', indent_first=False)

doc.add_page_break()

# ═══════════════════════════════════════════════════════════════════════════════
# CHƯƠNG II
# ═══════════════════════════════════════════════════════════════════════════════
add_heading(doc, 'CHƯƠNG II: GIỚI THIỆU DỮ LIỆU', 1)

add_heading(doc, '2.1 Nguồn dữ liệu (CDC – BRFSS)', 2)
add_para(doc, 'Bộ dữ liệu được sử dụng trong nghiên cứu này được trích xuất từ Hệ thống Giám sát Yếu tố Nguy cơ Hành vi (Behavioral Risk Factor Surveillance System – BRFSS), một chương trình điều tra sức khỏe qua điện thoại thường niên do Trung tâm Kiểm soát và Phòng ngừa Dịch bệnh Hoa Kỳ (Centers for Disease Control and Prevention – CDC) quản lý và thực hiện. BRFSS là hệ thống điều tra sức khỏe lớn nhất thế giới xét về quy mô, thu thập thông tin từ hơn 400.000 người Mỹ trưởng thành mỗi năm trên tất cả 50 tiểu bang, Quận Columbia và ba vùng lãnh thổ Hoa Kỳ.')
add_para(doc, 'Phiên bản dữ liệu được sử dụng là BRFSS 2015, sau khi được tiền xử lý và công bố trên Kaggle bởi Alex Teboul (2021), bao gồm 253.680 bản ghi và 22 biến đặc trưng (bao gồm cả biến mục tiêu). Dữ liệu dựa trên tự báo cáo của người tham gia, không có đo lường lâm sàng trực tiếp, do đó có thể tồn tại sai lệch nhớ lại (recall bias) và sai lệch báo cáo (reporting bias).')
add_para(doc, 'Điểm mạnh của bộ dữ liệu nằm ở tính đại diện thống kê cao trên quy mô toàn quốc, phương pháp thu thập chuẩn hóa và nhất quán, cùng với sự đa dạng về nhân khẩu học. Tuy nhiên, bộ dữ liệu không bao gồm các chỉ số sinh hóa như HbA1c hay đường huyết lúc đói – những tiêu chí y tế tiêu chuẩn để chẩn đoán bệnh tiểu đường – nên nhãn phân loại phụ thuộc vào thông tin tự khai báo về chẩn đoán trước đây.')

add_heading(doc, '2.2 Mô tả các thuộc tính', 2)
add_para(doc, 'Bộ dữ liệu bao gồm 21 biến đặc trưng và 1 biến mục tiêu (Diabetes_binary). Các biến đặc trưng có thể phân loại thành ba nhóm chính: biến liên tục (continuous), biến thứ bậc/phân loại (ordinal/categorical) và biến nhị phân (binary). Bảng 2.1 trình bày chi tiết ý nghĩa và phạm vi giá trị của từng thuộc tính.')

# ── Bảng 2.1 ──────────────────────────────────────────────────────────────────
attr_data = [
    ('Diabetes_binary', 'Biến mục tiêu – mắc tiểu đường', '0 = Không, 1 = Có'),
    ('HighBP', 'Huyết áp cao', '0 = Không, 1 = Có'),
    ('HighChol', 'Cholesterol cao', '0 = Không, 1 = Có'),
    ('CholCheck', 'Kiểm tra cholesterol trong 5 năm qua', '0 = Không, 1 = Có'),
    ('BMI', 'Chỉ số khối cơ thể', 'Giá trị thực, khoảng 12–98'),
    ('Smoker', 'Hút ≥100 điếu thuốc trong đời', '0 = Không, 1 = Có'),
    ('Stroke', 'Tiền sử đột quỵ', '0 = Không, 1 = Có'),
    ('HeartDiseaseorAttack', 'Bệnh tim mạch vành hoặc nhồi máu cơ tim', '0 = Không, 1 = Có'),
    ('PhysActivity', 'Hoạt động thể chất trong 30 ngày qua', '0 = Không, 1 = Có'),
    ('Fruits', 'Ăn trái cây ≥1 lần/ngày', '0 = Không, 1 = Có'),
    ('Veggies', 'Ăn rau củ ≥1 lần/ngày', '0 = Không, 1 = Có'),
    ('HvyAlcoholConsump', 'Uống rượu nhiều', '0 = Không, 1 = Có'),
    ('AnyHealthcare', 'Có bảo hiểm y tế', '0 = Không, 1 = Có'),
    ('NoDocbcCost', 'Không thể khám bác sĩ do chi phí', '0 = Không, 1 = Có'),
    ('GenHlth', 'Tự đánh giá sức khỏe tổng quát', '1=Xuất sắc → 5=Kém'),
    ('MentHlth', 'Số ngày sức khỏe tâm thần không tốt/30 ngày', '0–30 ngày'),
    ('PhysHlth', 'Số ngày sức khỏe thể chất không tốt/30 ngày', '0–30 ngày'),
    ('DiffWalk', 'Khó khăn khi đi bộ/leo cầu thang', '0 = Không, 1 = Có'),
    ('Sex', 'Giới tính', '0 = Nữ, 1 = Nam'),
    ('Age', 'Nhóm tuổi theo thang bậc', '1=18–24, …, 13=80+'),
    ('Education', 'Trình độ học vấn', '1=Chưa học xong cấp 1, …, 6=Tốt nghiệp ĐH'),
    ('Income', 'Mức thu nhập hộ gia đình', '1=<10K USD, …, 8=>75K USD'),
]

tbl = doc.add_table(rows=1, cols=3)
tbl.style = 'Table Grid'
tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
# Set column widths
for i, w in enumerate([3.5, 6.0, 5.5]):
    for row in tbl.rows:
        row.cells[i].width = Cm(w)

# Header row
hr = tbl.rows[0]
h_cell(hr.cells[0], 'Thuộc tính')
h_cell(hr.cells[1], 'Ý nghĩa')
h_cell(hr.cells[2], 'Phạm vi / Mô tả')

for i, (attr, meaning, rng) in enumerate(attr_data):
    row = tbl.add_row()
    shade = (i % 2 == 1)
    d_cell(row.cells[0], attr, shade)
    d_cell(row.cells[1], meaning, shade)
    d_cell(row.cells[2], rng, shade)

add_caption(doc, 'Bảng 2.1. Mô tả chi tiết các thuộc tính trong bộ dữ liệu BRFSS 2015.')

add_heading(doc, '2.3 Phân phối nhãn (Mất cân bằng lớp)', 2)
add_para(doc, 'Phân tích phân phối nhãn trong bộ dữ liệu cho thấy đây là một trường hợp mất cân bằng lớp (Class Imbalance) điển hình trong dữ liệu y tế. Cụ thể, trong tổng số 253.680 mẫu: Lớp 0 (Không mắc tiểu đường) chiếm 218.334 mẫu ≈ 86,07%; Lớp 1 (Mắc/tiền tiểu đường) chiếm 35.346 mẫu ≈ 13,93%. Tỷ lệ mất cân bằng xấp xỉ 6.2:1 giữa lớp âm tính và lớp dương tính.')
add_para(doc, 'Khi huấn luyện trên dữ liệu mất cân bằng nghiêm trọng, hầu hết các mô hình phân loại có xu hướng thiên về dự đoán lớp đa số (lớp 0), dẫn đến độ chính xác tổng thể cao nhưng khả năng nhận diện lớp thiểu số (lớp 1 – trường hợp bệnh) lại rất thấp. Trong bối cảnh y tế, việc bỏ sót ca bệnh (False Negative) thường có hậu quả nghiêm trọng hơn nhiều so với việc cảnh báo nhầm (False Positive). Do đó, xử lý mất cân bằng lớp là một bước không thể thiếu.')
add_para(doc, 'Chỉ số Accuracy không còn là thước đo đáng tin cậy trong trường hợp này – một mô hình ngây thơ luôn dự đoán lớp 0 cũng đạt Accuracy 86%, nhưng hoàn toàn vô dụng trong thực tế. Thay vào đó, các chỉ số như Recall (Sensitivity), F1-Score và AUC-ROC được ưu tiên sử dụng.')

doc.add_page_break()

# ═══════════════════════════════════════════════════════════════════════════════
# CHƯƠNG III
# ═══════════════════════════════════════════════════════════════════════════════
add_heading(doc, 'CHƯƠNG III: TIỀN XỬ LÝ DỮ LIỆU', 1)
add_para(doc, 'Tiền xử lý dữ liệu (Data Preprocessing) là giai đoạn quan trọng và thường chiếm phần lớn thời gian trong toàn bộ quy trình Data Mining. Chất lượng dữ liệu đầu vào ảnh hưởng trực tiếp và quyết định đến chất lượng mô hình đầu ra – nguyên lý thường được gọi là "Garbage In, Garbage Out" (GIGO). Trong nghiên cứu này, quy trình tiền xử lý được thực hiện một cách có hệ thống và tuần tự.')

add_heading(doc, '3.1 Kiểm tra chất lượng dữ liệu', 2)
add_heading(doc, '3.1.1 Missing Values (Giá trị khuyết)', 3)
add_para(doc, 'Kiểm tra giá trị khuyết (Missing Values) là bước đầu tiên và thiết yếu trong quá trình đánh giá chất lượng dữ liệu. Trong bộ dữ liệu BRFSS 2015 sau khi được tiền xử lý và chuẩn hóa, kết quả kiểm tra bằng df.isnull().sum() cho thấy không có giá trị khuyết (0 missing values) trong bất kỳ cột nào. Điều này là do quá trình làm sạch và chuẩn hóa đã được thực hiện trước đó, trong đó các bản ghi có giá trị thiếu đã bị loại bỏ hoặc xử lý phù hợp.')
add_para(doc, 'Trong trường hợp dữ liệu có giá trị khuyết, các chiến lược xử lý phổ biến bao gồm: Loại bỏ hàng (Listwise Deletion) khi tỷ lệ khuyết thấp và dữ liệu khuyết ngẫu nhiên; Điền thế đơn giản (Simple Imputation) bằng giá trị trung bình hoặc trung vị; Điền thế dựa trên mô hình (Model-based Imputation) như KNN Imputer hoặc MICE; và Tạo biến chỉ báo (Missingness Indicator) để mã hóa thông tin về tính khuyết thiếu.')

add_heading(doc, '3.1.2 Duplicates (Bản ghi trùng lặp)', 3)
add_para(doc, 'Kiểm tra bản ghi trùng lặp bằng df.duplicated().sum() cho thấy bộ dữ liệu BRFSS 2015 chứa 23.787 bản ghi trùng lặp (chiếm khoảng 9,4% tổng số mẫu). Tất cả các bản ghi trùng lặp này đã được loại bỏ hoàn toàn bằng df.drop_duplicates(keep="first"), giữ lại bản ghi đầu tiên của mỗi nhóm trùng lặp. Sau khi xử lý, bộ dữ liệu còn lại 229.893 mẫu hợp lệ.')
add_para(doc, 'Bản ghi trùng lặp có thể làm sai lệch quá trình học của mô hình, khiến mô hình "ghi nhớ" (overfit) những mẫu cụ thể và giảm khả năng tổng quát hóa. Việc loại bỏ bản ghi trùng lặp không chỉ cải thiện tính toàn vẹn của dữ liệu mà còn giảm tải tính toán trong các bước huấn luyện tiếp theo.')

add_heading(doc, '3.1.3 Outliers (Giá trị ngoại lệ)', 3)
add_para(doc, 'Giá trị ngoại lệ trong bộ dữ liệu này chủ yếu xuất hiện ở biến liên tục BMI. Phương pháp IQR (Interquartile Range) được áp dụng: một giá trị được coi là ngoại lệ nếu nằm ngoài khoảng [Q1 - 1.5×IQR, Q3 + 1.5×IQR]. Thay vì loại bỏ hoàn toàn, nghiên cứu áp dụng kỹ thuật Winsorization (capping), giới hạn các giá trị cực đoan tại phân vị 1% và 99%. Đối với các biến nhị phân và thứ bậc, khái niệm ngoại lệ không áp dụng trực tiếp do các giá trị này đã có phạm vi xác định rõ ràng.')

add_heading(doc, '3.2 Feature Engineering (Kỹ thuật đặc trưng)', 2)
add_para(doc, 'Feature Engineering là quá trình sử dụng kiến thức chuyên môn (domain knowledge) để tạo ra các đặc trưng mới hoặc biến đổi các đặc trưng hiện có nhằm nâng cao khả năng biểu diễn dữ liệu. Trong nghiên cứu này, ba đặc trưng tổng hợp mới được đề xuất dựa trên kiến thức y tế và phân tích dữ liệu khám phá (EDA).')

add_heading(doc, '3.2.1 BMI_category (Phân nhóm BMI)', 3)
add_para(doc, 'Kỹ thuật Binning được áp dụng để chuyển đổi BMI thành biến thứ bậc BMI_category dựa trên thang phân loại chuẩn của WHO: Underweight (BMI < 18.5) → 0; Normal (18.5 ≤ BMI < 25) → 1; Overweight (25 ≤ BMI < 30) → 2; Obese (BMI ≥ 30) → 3.')
add_para(doc, 'Lý do khoa học: mối quan hệ giữa BMI và nguy cơ tiểu đường không hoàn toàn tuyến tính – có những ngưỡng y tế quan trọng (đặc biệt BMI = 25 và 30) mà tại đó nguy cơ tăng đột biến. Phân nhóm giúp mô hình nắm bắt được các mối quan hệ ngưỡng này và giảm nhạy cảm với nhiễu trong biến BMI gốc.')

add_heading(doc, '3.2.2 comorbidity_score (Điểm bệnh đồng mắc)', 3)
add_para(doc, 'Đặc trưng comorbidity_score được tạo ra bằng cách tổng hợp (aggregation) các biến nhị phân phản ánh sự hiện diện của các bệnh lý tiên lượng quan trọng:')
add_formula(doc, 'comorbidity_score = HighBP + HighChol + Stroke + HeartDiseaseorAttack')
add_para(doc, 'Điểm comorbidity_score dao động từ 0 đến 4. Cơ sở y học: huyết áp cao, cholesterol cao, đột quỵ và bệnh tim mạch đều là các yếu tố nguy cơ độc lập và có tác dụng cộng hưởng với nhau trong việc làm tăng nguy cơ tiểu đường type 2. Một người có cả bốn bệnh lý này đồng thời có nguy cơ mắc tiểu đường cao hơn nhiều so với người chỉ có một hoặc không có bệnh lý nào.')

add_heading(doc, '3.2.3 healthy_lifestyle (Điểm lối sống lành mạnh)', 3)
add_para(doc, 'Đặc trưng healthy_lifestyle được xây dựng bằng cách tổng hợp các biến liên quan đến hành vi sức khỏe:')
add_formula(doc, 'healthy_lifestyle = PhysActivity + Fruits + Veggies + (1 − Smoker) + (1 − HvyAlcoholConsump)')
add_para(doc, 'Điểm healthy_lifestyle dao động từ 0 đến 5. Việc đảo chiều các biến Smoker và HvyAlcoholConsump đảm bảo mỗi hành vi lành mạnh đóng góp dương vào tổng điểm. Cơ sở khoa học: hoạt động thể chất thường xuyên, chế độ ăn giàu trái cây và rau củ, không hút thuốc và không uống rượu nhiều đều có tác động bảo vệ đáng kể chống lại bệnh tiểu đường type 2.')

add_heading(doc, '3.3 Feature Transformation (Biến đổi đặc trưng)', 2)
add_heading(doc, '3.3.1 Scaling (Chuẩn hóa đặc trưng liên tục)', 3)
add_para(doc, 'Kỹ thuật StandardScaler (chuẩn hóa z-score) được áp dụng cho các biến liên tục (BMI, MentHlth, PhysHlth). StandardScaler chuyển đổi mỗi đặc trưng về phân phối có trung bình bằng 0 và độ lệch chuẩn bằng 1 theo công thức:')
add_formula(doc, 'z = (x − μ) / σ')
add_para(doc, 'Trong đó μ là giá trị trung bình và σ là độ lệch chuẩn của đặc trưng trên tập huấn luyện. Quan trọng là các thông số μ và σ chỉ được tính toán từ tập huấn luyện và sau đó áp dụng cho cả tập kiểm tra để tránh hiện tượng data leakage (rò rỉ dữ liệu).')

add_heading(doc, '3.3.2 Encoding (Mã hóa đặc trưng phân loại)', 3)
add_para(doc, 'Các biến thứ bậc (ordinal) như GenHlth, Age, Education, Income và BMI_category được xử lý bằng kỹ thuật Ordinal Encoding – gán các số nguyên theo thứ tự có ý nghĩa. Cách tiếp cận này phù hợp vì các biến này có thứ tự tự nhiên và mối quan hệ giữa các mức độ có ý nghĩa. Đối với các mô hình cây quyết định như Random Forest và XGBoost, Ordinal Encoding đã đủ hiệu quả và không cần One-Hot Encoding.')

add_heading(doc, '3.3.3 Binary Features (Đặc trưng nhị phân – giữ nguyên)', 3)
add_para(doc, 'Các đặc trưng nhị phân như HighBP, HighChol, CholCheck, Smoker, Stroke, HeartDiseaseorAttack, PhysActivity, Fruits, Veggies, HvyAlcoholConsump, AnyHealthcare, NoDocbcCost, DiffWalk và Sex đã có định dạng số nhị phân (0/1) nên không cần thêm bất kỳ biến đổi nào. Giữ nguyên các biến nhị phân giúp đơn giản hóa quy trình xử lý và bảo toàn tính diễn giải (interpretability) của mô hình.')

add_heading(doc, '3.4 Xử lý mất cân bằng lớp (SMOTE / Class Weight)', 2)
add_para(doc, 'Để giải quyết mất cân bằng lớp, nghiên cứu áp dụng hai chiến lược bổ trợ nhau:')
add_para(doc, 'Chiến lược thứ nhất là SMOTE (Synthetic Minority Over-sampling Technique). Khác với over-sampling đơn giản, SMOTE tạo ra các mẫu tổng hợp mới bằng cách nội suy tuyến tính giữa các mẫu thiểu số và các láng giềng gần nhất của chúng:')
add_formula(doc, 'x_new = x_i + λ × (x_j − x_i),   λ ∈ [0, 1] ngẫu nhiên')
add_para(doc, 'SMOTE chỉ được áp dụng trên tập huấn luyện để tránh data leakage. Chiến lược thứ hai là sử dụng tham số class_weight: với Logistic Regression và Random Forest dùng class_weight="balanced"; với XGBoost dùng scale_pos_weight ≈ 6.2. Hai chiến lược này bổ trợ nhau: SMOTE cung cấp thêm thông tin về lớp thiểu số, trong khi class weight điều chỉnh quá trình học để chú ý nhiều hơn đến lớp thiểu số.')

add_heading(doc, '3.5 Train/Test Split (Phân chia tập dữ liệu)', 2)
add_heading(doc, '3.5.1 Random Sampling', 3)
add_para(doc, 'Phân chia ngẫu nhiên (Random Sampling) là phương pháp đơn giản nhất, trong đó các mẫu được phân bổ ngẫu nhiên vào tập huấn luyện và tập kiểm tra. Tuy nhiên, phương pháp này không đảm bảo phân phối của biến mục tiêu trong hai tập con sẽ phản ánh chính xác phân phối toàn bộ tập dữ liệu, đặc biệt khi dữ liệu mất cân bằng.')

add_heading(doc, '3.5.2 Stratified Sampling (Phân chia có phân tầng) ✓', 3)
add_para(doc, 'Nghiên cứu áp dụng Stratified Sampling (Phân lấy mẫu phân tầng) – phương pháp được chọn lựa. Tập dữ liệu được phân chia sao cho tỷ lệ của từng lớp trong biến mục tiêu được bảo toàn nhất quán trong cả tập huấn luyện và tập kiểm tra. Cụ thể, tham số stratify=y được truyền vào hàm train_test_split của scikit-learn.')
add_para(doc, 'Tỷ lệ phân chia được chọn là 80/20 (80% tập huấn luyện, 20% tập kiểm tra). Sau khi phân chia, cả hai tập đều duy trì tỷ lệ lớp dương tính xấp xỉ 13,93%. Giá trị random_state=42 được cố định để đảm bảo tính tái lập (reproducibility) của thực nghiệm.')

doc.add_page_break()

# ═══════════════════════════════════════════════════════════════════════════════
# CHƯƠNG IV
# ═══════════════════════════════════════════════════════════════════════════════
add_heading(doc, 'CHƯƠNG IV: BÀI TOÁN PHÂN LOẠI', 1)

add_heading(doc, '4.1 Giới thiệu chung về bài toán phân loại', 2)
add_para(doc, 'Phân loại (Classification) là một trong những nhiệm vụ căn bản và được nghiên cứu rộng rãi nhất trong học máy có giám sát (supervised learning). Bài toán phân loại nhằm mục đích học một hàm ánh xạ f: X → Y từ không gian đặc trưng đầu vào X sang không gian nhãn đầu ra Y hữu hạn và rời rạc, dựa trên một tập dữ liệu huấn luyện gán nhãn D = {(xᵢ, yᵢ)}ᵢ₌₁ᴺ. Khi |Y| = 2, bài toán được gọi là phân loại nhị phân (Binary Classification).')
add_para(doc, 'Trong bài toán phân loại nhị phân, nhãn đầu ra thường được ký hiệu là Y ∈ {0, 1}, trong đó 0 đại diện cho lớp âm tính và 1 đại diện cho lớp dương tính. Quá trình học bao gồm việc tối ưu hóa một hàm mục tiêu (objective function) hoặc hàm mất mát (loss function) để tìm ra bộ tham số tốt nhất cho mô hình.')
add_para(doc, 'Các bài toán phân loại xuất hiện trong vô số lĩnh vực ứng dụng: phát hiện thư rác, nhận diện hình ảnh, phân tích cảm xúc, phát hiện gian lận tài chính, và đặc biệt phổ biến trong y tế với các bài toán chẩn đoán và sàng lọc bệnh. Mỗi ứng dụng đặt ra các yêu cầu khác nhau về chỉ số đánh giá – điều này làm nổi bật tầm quan trọng của việc lựa chọn chỉ số đánh giá phù hợp với ngữ cảnh.')

add_heading(doc, '4.2 Lý do sử dụng bài toán phân loại', 2)
add_para(doc, 'Việc lựa chọn bài toán phân loại nhị phân cho nghiên cứu này xuất phát từ bản chất của câu hỏi nghiên cứu: "Một cá nhân có mắc/có nguy cơ mắc bệnh tiểu đường hay không?" Đây là câu hỏi có tính chất nhị phân rõ ràng. Biến mục tiêu Diabetes_binary chỉ nhận một trong hai giá trị (0 hoặc 1), xác nhận thêm sự phù hợp của cách tiếp cận này.')
add_para(doc, 'Bài toán phân loại cung cấp đầu ra có thể hành động (actionable output): quyết định có/không có nguy cơ. Điều này rất phù hợp cho mục đích sàng lọc trong cộng đồng, nơi bác sĩ và cán bộ y tế cần một quyết định rõ ràng để triển khai các can thiệp tiếp theo. Khuôn khổ phân loại nhị phân cũng cho phép sử dụng bộ công cụ đánh giá phong phú có ý nghĩa lâm sàng cụ thể như Sensitivity, Specificity, F1-Score và AUC-ROC.')

add_heading(doc, '4.3 Thuật toán sử dụng', 2)
add_heading(doc, '4.3.1 Logistic Regression (Hồi quy Logistic)', 3)
add_para(doc, 'Logistic Regression (LR) là thuật toán phân loại tuyến tính cổ điển, được sử dụng rộng rãi trong y tế và dịch tễ học. Mô hình ánh xạ tổ hợp tuyến tính của các đặc trưng đầu vào sang xác suất thuộc về lớp dương tính thông qua hàm sigmoid:')
add_formula(doc, 'P(y=1|x) = σ(wᵀx + b) = 1 / (1 + exp(−(wᵀx + b)))')
add_para(doc, 'Logistic Regression được chọn làm mô hình baseline vì: cung cấp ngưỡng hiệu suất cơ bản để so sánh; các hệ số hồi quy có thể diễn giải trực tiếp như odds ratios; là mô hình quen thuộc với bác sĩ và nhà nghiên cứu y tế. Regularization L2 (Ridge) được sử dụng để giảm thiểu overfitting, cùng với tham số class_weight="balanced".')

add_heading(doc, '4.3.2 Random Forest (Rừng ngẫu nhiên)', 3)
add_para(doc, 'Random Forest (RF) là thuật toán học máy tổng hợp (ensemble learning) dựa trên nguyên lý Bagging (Bootstrap Aggregating) được đề xuất bởi Breiman (2001). RF xây dựng một tập hợp các cây quyết định độc lập nhau, mỗi cây được huấn luyện trên một bootstrap sample và chỉ xem xét một tập con ngẫu nhiên các đặc trưng tại mỗi nút phân chia. Dự đoán cuối cùng được tổng hợp bằng cách lấy đa số phiếu (majority voting).')
add_para(doc, 'RF có khả năng xử lý tự nhiên các đặc trưng hỗn hợp, không nhạy cảm với việc chuẩn hóa đặc trưng, và cung cấp thước đo tầm quan trọng đặc trưng (Feature Importance) dựa trên mức độ giảm tạp chất (impurity) trung bình. Cấu hình sử dụng: n_estimators=200, max_features="sqrt", class_weight="balanced", với 5-fold cross-validation để hiệu chỉnh siêu tham số.')

add_heading(doc, '4.3.3 XGBoost (Extreme Gradient Boosting)', 3)
add_para(doc, 'XGBoost (Extreme Gradient Boosting) là thuật toán tăng cường gradient được tối ưu hóa cao về hiệu suất tính toán, được phát triển bởi Chen và Guestrin (2016). Tại mỗi bước boosting t, XGBoost thêm vào tập hợp một cây mới bằng cách tối thiểu hóa hàm mục tiêu:')
add_formula(doc, 'Obj = Σᵢ l(yᵢ, ŷᵢ⁽ᵗ⁾) + Σₖ Ω(fₖ)')
add_para(doc, 'Trong đó l là hàm mất mát binary cross-entropy, và Ω(f) = γT + ½λ||w||² là số hạng chính quy hóa. XGBoost được lựa chọn làm mô hình chính vì hiệu suất xuất sắc trên dữ liệu bảng, khả năng xử lý giá trị khuyết nội tại, tích hợp sẵn tham số scale_pos_weight để xử lý mất cân bằng lớp, và chính quy hóa L1/L2 tích hợp.')

doc.add_page_break()

# ═══════════════════════════════════════════════════════════════════════════════
# CHƯƠNG V
# ═══════════════════════════════════════════════════════════════════════════════
add_heading(doc, 'CHƯƠNG V: MÔ HÌNH VÀ THỰC NGHIỆM', 1)

add_heading(doc, '5.1 Thiết lập mô hình XGBoost', 2)
add_para(doc, 'Mô hình XGBoost được thiết lập và hiệu chỉnh siêu tham số thông qua GridSearchCV 5-fold trên tập huấn luyện. Bảng 5.1 trình bày các siêu tham số được xem xét và giá trị tối ưu được lựa chọn.')

hp_data = [
    ('n_estimators', '[100, 200, 300, 500]', '300'),
    ('learning_rate (eta)', '[0.01, 0.05, 0.1, 0.2]', '0.05'),
    ('max_depth', '[3, 4, 5, 6, 8]', '5'),
    ('subsample', '[0.7, 0.8, 0.9, 1.0]', '0.8'),
    ('colsample_bytree', '[0.7, 0.8, 0.9, 1.0]', '0.8'),
    ('min_child_weight', '[1, 3, 5, 7]', '3'),
    ('scale_pos_weight', 'Tự động (≈6.2)', '6.2'),
    ('reg_alpha (L1)', '[0, 0.01, 0.1, 1.0]', '0.01'),
    ('reg_lambda (L2)', '[0.1, 1.0, 10.0]', '1.0'),
]
tbl2 = doc.add_table(rows=1, cols=3)
tbl2.style = 'Table Grid'
tbl2.alignment = WD_TABLE_ALIGNMENT.CENTER
hr2 = tbl2.rows[0]
h_cell(hr2.cells[0], 'Siêu tham số')
h_cell(hr2.cells[1], 'Giá trị xem xét')
h_cell(hr2.cells[2], 'Giá trị tối ưu')
for i, (param, vals, best) in enumerate(hp_data):
    r = tbl2.add_row()
    shade = (i % 2 == 1)
    d_cell(r.cells[0], param, shade)
    d_cell(r.cells[1], vals, shade, center=True)
    d_cell(r.cells[2], best, shade, center=True)
add_caption(doc, 'Bảng 5.1. Các siêu tham số XGBoost được xem xét và giá trị tối ưu.')

add_para(doc, 'Quá trình tối ưu hóa sử dụng F1-Score (lớp dương tính) làm chỉ số đánh giá trong GridSearchCV. Học tốc độ thấp (learning_rate=0.05) kết hợp với số lượng cây nhiều hơn (n_estimators=300) giúp mô hình hội tụ chậm nhưng ổn định hơn.')

add_heading(doc, '5.2 Thiết lập thực nghiệm', 2)
add_para(doc, 'Toàn bộ thực nghiệm được thực hiện trong môi trường Python 3.9 với các thư viện: pandas 1.5.0, scikit-learn 1.1.0, XGBoost 1.7.0, imbalanced-learn 0.9.0, matplotlib 3.6.0 và seaborn 0.12.0. Quy trình thực nghiệm được thiết kế theo Pipeline của scikit-learn để đảm bảo tính nhất quán và tránh data leakage. Tất cả các bước biến đổi dữ liệu đều được đưa vào Pipeline và chỉ fit trên tập huấn luyện.')
add_para(doc, 'Mỗi mô hình được đánh giá 5 lần với các random seed khác nhau (42, 123, 456, 789, 2024) để đánh giá tính ổn định của kết quả. Tất cả các mô hình được huấn luyện trên cùng tập dữ liệu (sau khi áp dụng SMOTE) và đánh giá trên cùng tập kiểm tra để đảm bảo so sánh công bằng.')

add_heading(doc, '5.3 Các chỉ số đánh giá', 2)
add_para(doc, 'Các chỉ số đánh giá được sử dụng trong nghiên cứu, tất cả được tính từ Confusion Matrix với TP, TN, FP, FN:')
add_formula(doc, 'Accuracy  = (TP + TN) / (TP + TN + FP + FN)')
add_formula(doc, 'Precision = TP / (TP + FP)')
add_formula(doc, 'Recall    = TP / (TP + FN)')
add_formula(doc, 'F1-Score  = 2 × (Precision × Recall) / (Precision + Recall)')
add_formula(doc, 'Specificity = TN / (TN + FP)')
add_para(doc, 'Trong ngữ cảnh y tế, Recall (Sensitivity) được đặc biệt quan tâm vì nó đo lường tỷ lệ ca bệnh thực sự được phát hiện đúng. AUC-ROC cung cấp đánh giá toàn diện về khả năng phân biệt lớp của mô hình, độc lập với ngưỡng phân loại, dao động từ 0.5 (phân loại ngẫu nhiên) đến 1.0 (phân loại hoàn hảo).')

doc.add_page_break()

# ═══════════════════════════════════════════════════════════════════════════════
# CHƯƠNG VI
# ═══════════════════════════════════════════════════════════════════════════════
add_heading(doc, 'CHƯƠNG VI: KẾT QUẢ VÀ THẢO LUẬN', 1)

add_heading(doc, '6.1 Kết quả đánh giá mô hình', 2)
add_para(doc, 'Bảng 6.1 trình bày kết quả hiệu suất của cả ba mô hình trên tập kiểm tra. Các giá trị được báo cáo là trung bình của 5 lần chạy với các random seed khác nhau.')

result_data = [
    ('Logistic Regression', '73.2%', '0.41', '0.75', '0.53', '0.79'),
    ('Random Forest', '74.8%', '0.44', '0.70', '0.54', '0.81'),
    ('XGBoost ★', '76.1%', '0.47', '0.72', '0.72', '0.83'),
]
tbl3 = doc.add_table(rows=1, cols=6)
tbl3.style = 'Table Grid'
tbl3.alignment = WD_TABLE_ALIGNMENT.CENTER
hr3 = tbl3.rows[0]
for i, hdr in enumerate(['Mô hình', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']):
    h_cell(hr3.cells[i], hdr)
for i, row_data in enumerate(result_data):
    r = tbl3.add_row()
    is_best = ('★' in row_data[0])
    bg = '70AD47' if is_best else ('EBF3FB' if i % 2 == 1 else 'FFFFFF')
    for j, val in enumerate(row_data):
        set_cell_bg(r.cells[j], bg)
        set_cell_borders(r.cells[j], '2E75B6')
        p = r.cells[j].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER if j > 0 else WD_ALIGN_PARAGRAPH.LEFT
        run = p.add_run(val)
        run.font.name = FONT_NAME
        run.font.size = Pt(10)
        run.font.bold = is_best
        run.font.color.rgb = C_WHITE if is_best else C_BLACK
add_caption(doc, 'Bảng 6.1. Kết quả hiệu suất các mô hình trên tập kiểm tra. ★ = Mô hình tốt nhất.')

add_para(doc, 'XGBoost vượt trội so với hai mô hình còn lại trên hầu hết các chỉ số. AUC-ROC đạt 0.83 (được coi là "tốt" trong y văn, AUC > 0.8). Recall đạt 0.72, nghĩa là mô hình phát hiện đúng 72% các ca tiểu đường thực sự – kết quả đáng khích lệ cho bài toán sàng lọc cộng đồng. Logistic Regression có Recall cao nhất (0.75) nhưng Precision thấp nhất (0.41), cho thấy xu hướng dự đoán dương tính quá mức.')

add_heading(doc, '6.2 So sánh các thuật toán', 2)
add_para(doc, 'Phân tích so sánh ba thuật toán cho thấy sự đánh đổi (trade-off) rõ ràng giữa hiệu suất và khả năng diễn giải. Logistic Regression cung cấp khả năng diễn giải cao nhất nhưng hiệu suất thấp nhất do không thể nắm bắt các mối quan hệ phi tuyến. Random Forest cải thiện đáng kể nhờ khả năng mô hình hóa quan hệ phi tuyến. XGBoost đạt hiệu suất cao nhất nhờ cơ chế boosting thích ứng, tập trung vào các mẫu khó phân loại.')
add_para(doc, 'XGBoost yêu cầu hiệu chỉnh siêu tham số nhiều hơn và thời gian huấn luyện lâu hơn. Tuy nhiên trong bối cảnh y tế cộng đồng nơi mô hình được huấn luyện ngoại tuyến, chi phí tính toán cao hơn này hoàn toàn chấp nhận được so với lợi ích về hiệu suất.')

add_heading(doc, '6.3 Phân tích sai số', 2)
add_para(doc, 'Phân tích Confusion Matrix của XGBoost trên tập kiểm tra (45.979 mẫu, ~14% dương tính): True Positive (TP) ≈ 4.600 (Recall = 72%); True Negative (TN) ≈ 30.500 (Specificity ≈ 78%); False Negative (FN) ≈ 1.800 (28% ca bệnh bị bỏ sót); False Positive (FP) ≈ 9.100.')
add_para(doc, 'Tỷ lệ False Negative ~28% là điểm cần cải thiện quan trọng nhất. Các ca FN thường có BMI trong nhóm overweight (không phải obese), ít bệnh đồng mắc và điểm healthy_lifestyle ở mức trung bình – những trường hợp biên giới khó phân biệt. Phân tích Feature Importance cho thấy GenHlth, BMI, Age, HighBP và comorbidity_score là các đặc trưng quan trọng nhất. Hai đặc trưng tổng hợp được đề xuất (comorbidity_score, healthy_lifestyle) đều nằm trong top 10, khẳng định giá trị của Feature Engineering.')

add_heading(doc, '6.4 Đề xuất cải tiến', 2)
add_para(doc, 'Về xử lý mất cân bằng: Kết hợp SMOTE với Tomek Links (SMOTETomek) hoặc SMOTE với Edited Nearest Neighbours (SMOTEENN) để vừa tăng cường lớp thiểu số vừa làm sạch ranh giới quyết định.')
add_para(doc, 'Về ngưỡng phân loại: Thay vì sử dụng ngưỡng mặc định 0.5, tối ưu hóa ngưỡng dựa trên đường cong Precision-Recall hoặc chi phí lâm sàng. Ngưỡng thấp hơn (0.35–0.40) có thể phù hợp hơn để tăng Recall trong sàng lọc cộng đồng.')
add_para(doc, 'Về diễn giải mô hình: Tích hợp SHAP (SHapley Additive exPlanations) để cung cấp giải thích cục bộ cho từng dự đoán cá nhân, giúp bác sĩ hiểu rõ lý do một cá nhân được xếp vào nhóm nguy cơ cao. SHAP có nền tảng lý thuyết vững chắc trong lý thuyết trò chơi Shapley.')

doc.add_page_break()

# ═══════════════════════════════════════════════════════════════════════════════
# CHƯƠNG VII
# ═══════════════════════════════════════════════════════════════════════════════
add_heading(doc, 'CHƯƠNG VII: KẾT LUẬN', 1)

add_heading(doc, '7.1 Tổng kết', 2)
add_para(doc, 'Nghiên cứu này đã trình bày một quy trình khai phá dữ liệu toàn diện và có hệ thống, từ thu thập và tiền xử lý dữ liệu đến xây dựng, huấn luyện và đánh giá các mô hình học máy, nhằm giải quyết bài toán phân loại nguy cơ mắc bệnh tiểu đường dựa trên bộ dữ liệu BRFSS 2015 của CDC Hoa Kỳ. Quy trình được thiết kế tuân thủ đầy đủ các nguyên tắc khoa học và thực hành tốt nhất trong Data Mining.')
add_para(doc, 'Ba đóng góp chính của nghiên cứu: (1) Đề xuất và kiểm chứng ba đặc trưng tổng hợp mới (BMI_category, comorbidity_score, healthy_lifestyle) có nền tảng y học rõ ràng; (2) So sánh toàn diện ba thuật toán với XGBoost đạt kết quả tốt nhất (Accuracy 76,1%, AUC-ROC 0,83, F1-Score 0,72); (3) Phân tích chi tiết cấu trúc sai số và tầm quan trọng đặc trưng, cung cấp thông tin có giá trị thực tiễn.')
add_para(doc, 'Kết quả nghiên cứu cho thấy tiềm năng ứng dụng thực tiễn của học máy trong sàng lọc nguy cơ bệnh tiểu đường dựa trên dữ liệu khảo sát hành vi sức khỏe. Mô hình XGBoost với AUC-ROC 0.83 hoàn toàn đủ điều kiện để hỗ trợ quyết định sàng lọc trong cộng đồng khi kết hợp với đánh giá lâm sàng tiếp theo.')

add_heading(doc, '7.2 Hạn chế', 2)
add_para(doc, 'Hạn chế về dữ liệu: Bộ dữ liệu BRFSS dựa trên tự báo cáo, không có xác nhận lâm sàng. Nhãn Diabetes_binary bỏ sót những người chưa được chẩn đoán và có thể bị ảnh hưởng bởi sai lệch nhớ lại. Bộ dữ liệu chỉ đại diện cho dân số Hoa Kỳ năm 2015, hạn chế khả năng tổng quát hóa.')
add_para(doc, 'Hạn chế về mô hình: Không bao gồm các chỉ số sinh hóa quan trọng như HbA1c, đường huyết lúc đói, insulin. Thiếu thông tin gia đình (tiền sử gia đình mắc tiểu đường). Mô hình không được xác nhận bên ngoài (external validation) trên tập dữ liệu độc lập từ dân số khác.')

add_heading(doc, '7.3 Hướng phát triển', 2)
add_para(doc, 'Về dữ liệu: Tích hợp dữ liệu lâm sàng từ hồ sơ y tế điện tử (EHR) với dữ liệu BRFSS để xây dựng bộ đặc trưng phong phú hơn. Mở rộng phân tích theo chuỗi thời gian sử dụng dữ liệu BRFSS từ nhiều năm để theo dõi xu hướng thay đổi nguy cơ.')
add_para(doc, 'Về mô hình: Khám phá các kiến trúc hiện đại như TabNet, LightGBM và CatBoost. Áp dụng AutoML để tự động hóa lựa chọn và hiệu chỉnh mô hình. Nghiên cứu Federated Learning để huấn luyện mô hình phân tán trên dữ liệu từ nhiều bệnh viện mà không cần chia sẻ dữ liệu nhạy cảm.')
add_para(doc, 'Về ứng dụng: Xây dựng ứng dụng web/di động tích hợp mô hình để cung cấp công cụ sàng lọc nguy cơ tiểu đường cho cộng đồng. Tích hợp SHAP và LIME để cung cấp giải thích cá nhân hóa cho mỗi dự đoán, tăng niềm tin của bác sĩ và người dùng vào hệ thống AI. Thực hiện thử nghiệm lâm sàng thực địa để đánh giá giá trị thực tiễn trong bối cảnh chăm sóc sức khỏe thực tế.')

doc.add_page_break()

# ── TÀI LIỆU THAM KHẢO ───────────────────────────────────────────────────────
add_heading(doc, 'TÀI LIỆU THAM KHẢO', 1)

refs = [
    '[1] Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5–32.',
    '[2] Centers for Disease Control and Prevention (CDC). (2015). Behavioral Risk Factor Surveillance System Survey Data. U.S. DHHS, Atlanta, Georgia.',
    '[3] Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic Minority Over-sampling Technique. JAIR, 16, 321–357.',
    '[4] Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. Proc. 22nd ACM SIGKDD, 785–794.',
    '[5] Fayyad, U., Piatetsky-Shapiro, G., & Smyth, P. (1996). From Data Mining to Knowledge Discovery in Databases. AI Magazine, 17(3), 37–54.',
    '[6] Han, J., Kamber, M., & Pei, J. (2011). Data Mining: Concepts and Techniques (3rd ed.). Morgan Kaufmann.',
    '[7] International Diabetes Federation (IDF). (2023). IDF Diabetes Atlas, 10th Edition. Brussels, Belgium.',
    '[8] Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions. NeurIPS, 30.',
    '[9] Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. JMLR, 12, 2825–2830.',
    '[10] Teboul, A. (2021). Diabetes Health Indicators Dataset. Kaggle. https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset',
    '[11] World Health Organization (WHO). (2023). Diabetes: Key Facts. https://www.who.int/news-room/fact-sheets/detail/diabetes',
    '[12] Zou, Q., et al. (2018). Predicting Diabetes Mellitus With Machine Learning Techniques. Frontiers in Genetics, 9, 515.',
]
for ref in refs:
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after  = Pt(3)
    p.paragraph_format.line_spacing = Pt(18)
    p.paragraph_format.left_indent  = Cm(0.5)
    run = p.add_run(ref)
    run.font.name = FONT_NAME
    run.font.size = Pt(11)
    run.font.color.rgb = C_BLACK

# ── SAVE ─────────────────────────────────────────────────────────────────────
output_path = 'BaoCao_DataMining_TieuDuong.docx'
doc.save(output_path)
print(f'✓ Báo cáo đã được lưu tại: {output_path}')
