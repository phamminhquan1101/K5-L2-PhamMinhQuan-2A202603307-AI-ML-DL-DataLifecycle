# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 06/11/2026

**Runtime Colab:** GPU-T4

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ            | Nghĩa ngắn                                                    |
| ------------- | ------------------------------------------------------------- |
| Prediction    | Kết quả model vừa đoán ra; chưa phải nhãn đúng                |
| Ground truth  | Nhãn do người gán theo quy tắc                                |
| Taxonomy      | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint    | File model `.pt` đã huấn luyện sẵn                            |
| Guideline     | Quy tắc gán nhãn do người viết                                |
| Annotator     | Người gán nhãn                                                |
| Reviewer / QC | Người kiểm tra nhãn                                           |
| Escalation    | Ca khó → hỏi Lab Coach, không tự bịa                          |
| Model score   | Số “tin” của model; không phải điểm chất lượng nhãn người     |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=468 ; `class_name`="cab" ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`="ImageNet-1K"
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
   Record nói về cả bức ảnh. Nó chỉ có class_id, class_name và score, không có bbox hay polygon nên không chỉ ra vật nào ở vị trí nào. "cab" là lớp model cho là phù hợp nhất với toàn ảnh, không có nghĩa trong ảnh chỉ có một chiếc taxi.
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
   Class list đi kèm checkpoint yolo11n-cls.pt, theo taxonomy ImageNet-1K (1000 lớp cố định). Model chỉ chọn trong danh sách này, không tự tạo ra tên lớp mới.
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
   Cùng một con số hay một cái tên có thể mang nghĩa khác nhau ở taxonomy khác nhau. Ghi đủ cả ba thì nhãn mới truy vết được, không bị nhầm khi gộp dữ liệu từ nhiều nguồn, và có thể đối chiếu khi đổi model.
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
   Guideline phải định nghĩa "vật chính" (vd. vật chiếm diện tích lớn nhất hoặc ở trung tâm), thứ tự ưu tiên giữa các lớp, và cách xử lý khi không chọn được: gắn ambiguous rồi hỏi Lab Coach, không tự đoán.
6. Vì sao không được lấy `score` của model làm ground truth?
   score = 0.51 chỉ là mức tự tin của model, không chứng minh "cab" là đúng. Ground truth phải do annotator gán theo guideline và reviewer kiểm tra. Nếu dùng prediction làm nhãn thì lỗi của model bị chép vào dữ liệu và không còn cách nào phát hiện ra.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`="person" ; `score`=0.912625 ; `bbox_xyxy`=[385.33,
   69.24, 498.92, 348.92] ; `bbox_width`=113.58 ; `bbox_height`=279.68
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
   Góc trên-trái của hộp ở (385.33, 69.24), tức cách mép trái ảnh khoảng 385 px và cách mép trên khoảng 69 px. Góc dưới-phải ở (498.92, 348.92). Hộp rộng 498.92 − 385.33 ≈ 113.6 px và cao 348.92 − 69.24 ≈ 279.7 px. Hộp cao gấp khoảng 2.5 lần chiều rộng, hợp với dáng một người đang đứng.
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
   Ở ngưỡng 0.35, ảnh kitchen có 11 prediction (2 person, 5 bowl, 2 oven, 2 cup). Ở ngưỡng 0.60 chỉ còn 6 (2 person, 2 bowl, 2 oven). 5 hộp bị loại đều là vật nhỏ có score thấp: 3 bowl (0.500, 0.465, 0.381) và 2 cup (0.451, 0.382). Hai cup đều bị loại hết. Ngưỡng cao giữ lại các hộp chắc chắn hơn nhưng có thể bỏ sót cốc và bát thật.
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
   Hạ ngưỡng thì giữ được nhiều vật thật hơn, nhưng cũng kéo theo nhiều hộp sai. Reviewer phải kiểm tra từng hộp có score thấp, xóa hộp không có vật thật hoặc hộp trùng, sửa class sai và chỉnh lại box lệch.
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
   Box phải ôm sát phần nhìn thấy được của vật: cả 4 cạnh chạm mép ngoài cùng của vật, không chừa nền thừa quá khoảng 2–3 px, và không cắt mất bộ phận nào. Mỗi vật đúng một box. Phần bị che khuất thì không đoán ra ngoài phần nhìn thấy.
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
   Tôi sẽ viết thành quy tắc trong guideline vì che khuất và cắt mép xảy ra ở rất nhiều ảnh, cần một cách làm thống nhất cho mọi annotator: chỉ vẽ phần nhìn thấy và box dừng ở mép ảnh. Nếu bị che nhiều đến mức không nhận ra là vật gì, tôi sẽ escalation lên Lab Coach thay vì tự đoán.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`="kitchen-001 ; `class_name`="person" ; `score`=0.899318 ; số điểm polygon=`polygon_point_count`=348 ; 3–5 điểm đầu của `polygon_xy`=[
   446.0,
   70.0
   ],
   [
   445.0,
   71.0
   ],
   [
   444.0,
   71.0
   ],
   [
   443.0,
   72.0
   ],
   [
   442.0,
   72.0
   ]
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
   Box chỉ cho biết vật nằm trong hình chữ nhật nào. Polygon cho biết đường viền thật của vật: dáng người, tay chân, phần lồi lõm.
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
   instance_id là mã riêng cho từng vật trong một ảnh, để phân biệt các vật cùng lớp (4 bowl có 4 id khác nhau). Nó không phải class ID vì không nói vật thuộc loại gì. Nó cũng không phải tracking ID vì không theo dõi cùng một vật qua nhiều ảnh hay frame. Chỉ là số thứ tự trong ảnh này.
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
   Mask chỉ bao phần nhìn thấy được của vật và đi theo mép ngoài cùng của vật, sai lệch không quá vài pixel. Không tô lan ra nền hoặc sang vật bên cạnh. Ở chỗ hai vật chạm nhau, mỗi pixel chỉ thuộc một instance, vật nằm trước được giữ phần chồng lấn. Lỗ hở thật bên trong vật (vd. quai cốc) thì không tô.
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
   Tôi viết thành guideline: "Hai vật chạm nhau thì tách biên theo đường tiếp giáp, mỗi pixel chỉ thuộc một instance." Đây là tình huống lặp lại rất nhiều (bát chồng bát, thìa cạnh thìa), cần một cách làm thống nhất.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ                | Ground truth trông như thế nào? (đơn vị nhãn)                                                                           | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON                                                                                                                                            | Annotator làm gì?                                                                                                                                       | Reviewer xem gì?                                                                                                                                                                        |
| --------------------- | ----------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Phân loại ảnh         | **1 nhãn cho cả ảnh**: `class_id` + `class_name` + `taxonomy_name` (vd. 468 / "cab" / ImageNet-1K). Không có tọa độ.    | Ảnh `traffic` có nhiều xe (JSON detection có nhiều bus, car) nhưng phải chọn 1 lớp; top-1 "cab" chỉ có score 0.51, model không chắc. Không rõ "vật chính" là gì.                              | Xem toàn ảnh, chọn **một** lớp trong taxonomy theo quy tắc "vật chính" của guideline; ảnh mơ hồ/không có lớp phù hợp thì gắn `ambiguous` và escalation. | Nhãn có đúng lớp và đúng taxonomy không; các annotator có chọn "vật chính" thống nhất không; ảnh `ambiguous` đã được xử lý chưa.                                                        |
| Phát hiện vật thể     | **1 box cho mỗi vật**: `class_name` + `bbox_xyxy` [x1, y1, x2, y2] (pixel). Vd. person [385.33, 69.24, 498.92, 348.92]. | Ảnh `kitchen`: 11 hộp ở ngưỡng 0.35 nhưng chỉ còn 6 ở 0.60; 3 bowl và 2 cup (score 0.38–0.50) bị loại, có thể là vật thật bị bỏ sót. Nhiều bowl nhỏ đặt sát nhau, dễ thiếu hoặc trùng hộp.    | Vẽ box **ôm sát phần nhìn thấy** cho **mọi** vật thuộc taxonomy, mỗi vật đúng 1 box, chọn đúng lớp; vật cắt mép thì box dừng ở mép ảnh.                 | Thiếu vật (miss), thừa hộp/hộp trùng, sai lớp (cup ↔ bowl), box lệch hoặc quá rộng; đặc biệt các hộp score thấp.                                                                        |
| Instance segmentation | **1 mask/polygon cho mỗi vật** + `instance_id` riêng + `class_name`. Vd. `kitchen-001` person, polygon 348 điểm.        | Hai thìa `kitchen-008` và `kitchen-011` sát nhau (box chạm ở x≈533); bowl `kitchen-002`/`006` chồng nhau trên `kitchen-005` dining table; `kitchen-009` person bị cắt ở mép trái, score 0.43. | Vẽ polygon theo **mép ngoài** phần nhìn thấy; tách biên ở chỗ 2 vật chạm nhau, mỗi pixel chỉ thuộc 1 instance; mỗi vật một `instance_id`.               | Mask có tràn ra nền hoặc sang vật bên cạnh không; các vật cùng lớp có bị gộp thành 1 instance không; biên chỗ chạm/che có đúng quy tắc không; mảnh nhỏ cắt mép có cần escalation không. |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
   Mình chỉ dùng các ảnh mẫu COCO do notebook cung cấp (có nguồn trong IMAGE_ATTRIBUTION.md), không thêm ảnh cá nhân hoặc ảnh ngoài phạm vi bài, và không ghi họ tên, MSSV, email hay bất kỳ dữ liệu cá nhân nào vào báo cáo hoặc output.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
   Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi (vd. ảnh có thông tin cá nhân hay ảnh nhạy cảm không thuộc bộ mẫu), mình sẽ dừng ngay, không xử lý, lưu hay đưa vào output, và báo cho Lab Coach / giảng viên phụ trách để được hướng dẫn, chứ không tự quyết định.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [x] `REPORT.md` (file này đã điền)
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [x] Đã nộp **link repository** trên LMS
