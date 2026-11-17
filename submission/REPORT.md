# Báo cáo Day 08 — 2A202603271 - Hà Huy Dũng

> Trong bài thực hành, tôi thử nghiệm mô hình, chọn ngưỡng, tổ chức xử lý nhãn trên CVAT, so sánh mask và kiểm tra chất lượng. Các kết quả dưới đây kèm mẫu số và nguồn đối chiếu. Máy thực hiện: RTX 3050 Laptop 4 GB (CUDA), CVAT v2.74.1 (Docker).

## 1. Model và recipe (Part 2–3)

| Lớp | Model + cỡ | Ngưỡng | Vì sao (dẫn `run_id` trong `model_log.csv`) |
|---|---|---|---|
| person / bicycle / motorbike / car / bus / truck | YOLO26-m, imgsz 1024 | 0.3 (mặc định) | Sweep trên gold (r12): **không lớp nào** đạt recall 90% với ngưỡng ≤ 0.5 (person tối đa 23/35 = 66%, motorbike 9/12 = 75%, car 5/8 = 62%, bicycle 5/6 = 83%) → script giữ mặc định. Ở 0.3 precision/recall: person 0.66/0.66, motorbike 0.73/0.67, car 0.63/0.63, bicycle 0.80/0.67. bus chỉ 1 vật (< 5), truck 0 vật → không chọn ngưỡng riêng. YOLOE kém ở lớp COCO (r05: 17 motorbike so với 55 của YOLO26 trên pilot). |
| three_wheeler | YOLOE-26l, 5 prompt: `tuk-tuk, cyclo, cargo tricycle, three-wheeled vehicle, auto rickshaw` | **0.5** | Sweep (r13): recall 7/7 = 100% ở mọi ngưỡng 0.1–0.5, precision 7/8 = 0.875 ở 0.3 và 0.5 nhưng chỉ 7/19 = 0.37 ở 0.1. Pilot (r04, r09) cho cùng kết luận: mọi box thừa (xe máy img_030 0.30/0.36, img_007 0.38, img_016 0.39; xe đạp img_015 0.30) đều < 0.4. Prompt mặc định 3 chữ (r03) sót hàng xích lô ở img_007 và 1 xích lô ở img_032 (r09 tìm đủ 4). |

- Lệnh `merge_prelabels.py` đã dùng:
  `python scripts/merge_prelabels.py work/prelabel/final_yoloel_c005/pre.coco.json work/prelabel/final_y26m_c005/pre.coco.json --thr "three_wheeler=0.5" --conf 0.3 --iou 0.6 --split "batch,gold" --out work/prelabel/merged_review/pre.coco.json`
  → 28 ảnh, 591 box (person 311, motorbike 153, car 65, three_wheeler 33, bicycle 14, truck 9, bus 6). Bẫy: `inject_traps.py --student 2A202603271`.
- **Phiên bản SAM chọn: SAM 2.1-b** (`sam2.1_b.pt`). Lý do đo được (r07, `work/mask_measure.py`): trên img_013 với 3 click âm, MobileSAM (SAM 1) sụp còn 23% diện tích box (mất nửa xe), SAM 2.1 giữ 41%; với box + 2 click âm, SAM 2.1 tách người lái tốt hơn (mask phủ 34% box người lái so với 40% của MobileSAM) và biên ít loang lổ hơn. Tốc độ gần bằng (1.72 vs 1.54 s/ảnh), 160 MB vừa GPU 4 GB, không cần quyền như SAM 3.
- **Lỗi không sửa được bằng ngưỡng:** YOLO26 gọi tuk-tuk là `car` (img_004 0.29, img_017 0.39, img_023 0.29), xích lô là `bicycle` (img_032 0.38), xe bán tải là `truck` (img_002 0.73), và **không ra box xe nào** cho tuk-tuk xanh ở img_002. Đây là lỗi **từ vựng** (80 lớp COCO không có xe ba bánh), nên tăng/giảm ngưỡng chỉ làm mất cả box. Sửa bằng model open-vocabulary (YOLOE) + merge cho `three_wheeler` thắng `car/truck` khi IoU ≥ 0.6, và còn lại thì người duyệt đổi lớp. Lỗi YOLOE **tách một tuk-tuk thành 2–3 box** (img_013, img_008) cũng không sửa bằng ngưỡng vì các box nửa xe có conf 0.83–0.97.
- **Điểm cao mà sai / điểm thấp mà đúng:** bán tải → `truck` 0.73 (img_002, sai lớp); box nửa xe tuk-tuk img_013 0.94 (sai hình); tuk-tuk → `car` 0.29 (img_004: đúng vật, sai lớp, điểm thấp); xích lô img_032 `three_wheeler` 0.27 (đúng, điểm rất thấp).
- **So sánh model mạnh hơn (đo thêm, không dùng cho lô vì pre-label đã nạp CVAT):** YOLO26-l imgsz 1280 trên gold ở ngưỡng 0.3 (`work/qc/sweep_y26l1280.*`): recall motorbike 10/12 = 0.83 (m: 0.67), car 7/8 = 0.88 (m: 0.63), person 24/35 = 0.69 (m: 0.66), nhưng precision giảm (person 0.52 vs 0.66, motorbike 0.53 vs 0.73) và chậm 2.8 s/ảnh vs 0.16. Lần sau sẽ dùng bản l-1280 cho motorbike/car (recall quan trọng hơn: xoá box thừa nhanh hơn vẽ box sót).
- **Giới hạn:** ngưỡng chọn và precision/recall chấm trên **cùng 4 ảnh gold** → số đo lạc quan; three_wheeler chỉ 7 vật, bus 1, truck 0 → ngưỡng là ước lượng thô. 12/35 người gold máy không bao giờ ra (nhỏ, ngồi trong xe, bị che), nên `added` ở batch sẽ không nhỏ.

## 2. Hồ sơ lỗi mô hình

Nguồn: `qc/review_stats.json` và `.md`, tính trên bản review lần đầu `labels/batch_reviewed.json`, trước sửa QC. Script loại 25 đối tượng gần vị trí bẫy khỏi thống kê chất lượng máy.

| Trạng thái | Số | Tỉ lệ | Mẫu số |
|---|---:|---:|---|
| accepted | 429 | 85,0% | 505 box máy |
| edited | 22 | 4,4% | 505 box máy |
| deleted | 50 | 9,9% | 505 box máy |
| added | 89 | 16,5% | 540 đối tượng sau duyệt |

Accepted + edited + deleted = 501, thấp hơn 505 vì 4 box không ghép nằm trong vùng ignore không được tính deleted, trong khi mẫu số của script vẫn giữ chúng. Có 8 trường hợp đổi lớp được tính deleted + added; số sót ước tính là 89 − 8 = 81, không phải cả 89 đều là bỏ sót.

`three_wheeler` cần can thiệp nhiều: 8 accepted, 7 edited, 9 deleted, 5 added; ví dụ một tuk-tuk bị tách thành hai nửa ở img_013, img_017. Xe đẩy hàng hai bánh ở img_008 còn bị nhận nhầm sau lần review đầu và chỉ được sửa trong QC. Person có 37 added, motorbike có 29 added, phản ánh khó khăn với người bị che và xe nhỏ.

So với 4 ảnh tham chiếu: **51 TP, 21 FP, 18 FN**, precision **51/72 = 70,8%**, recall **51/69 = 73,9%**. Mẫu số recall là 69 vật, không gồm 2 ignore_region. Tập tham chiếu không phải gold gán tay độc lập; ngưỡng còn được chọn trên chính tập này nên số đo không phải ước lượng tổng quát hóa.

## 3. Thiên kiến tự động hóa và ảnh bẫy

Nguồn lượt đầu: `qc/review_stats.md`, chấm sau khi đã xuất nhãn. Đáp án chỉ được dùng khi chấm sau export. Kết quả lượt đầu được giữ riêng với lượt kiểm tra lại dưới đây.

Điểm bắt bẫy được dùng để đánh giá lượt review đầu của quy trình đã thực hiện; phạm vi hỗ trợ được ghi ở mục 8.

- Bắt **8/12 = 66,7%**, thấp hơn yêu cầu ít nhất 10/12. Theo loại: loose 2/3, wrong_class 2/3, missing 1/3, extra 3/3.
- Bẫy lọt theo script: img_034 (wrong_class), img_026 (loose), img_018 và img_013 (missing).
- Accepted 85,0% nhưng bẫy chỉ 66,7% cho thấy giữ nhiều box không chứng minh review tốt. Việc tìm vật chưa có box yếu hơn phát hiện box thừa; cụm xe đỗ và phần người bị che cần lượt quét riêng.
- **Điểm cần cải thiện:** lượt review 24 ảnh mất 13,14 phút (0,55 phút/ảnh, nghỉ 0 phút). Trong bốn bẫy lọt có hai bẫy missing, một loose và một wrong_class. Việc bắt đủ 3/3 bẫy extra nhưng chỉ bắt 1/3 bẫy missing cho thấy quy trình phát hiện box thừa tốt hơn tìm vật chưa có box. Vì vậy, lần review tiếp theo cần một lượt quét ảnh khi ẩn nhãn, sau đó mới kiểm lớp và độ sát của từng box.
- Giữ nguyên điểm lần đầu để đánh giá quy trình. Các sửa QC không được dùng để thay điểm này bằng kết quả đẹp hơn. Khoảng tin cậy 95% của 8/12 rất rộng, khoảng 39–86%.

**Lượt kiểm tra lại sau rework:** tôi dùng một bộ 12 bẫy mới, seed `2026111701`, trên nhãn đã rà của cùng 24 ảnh. Nhãn được chốt trước khi chấm và mở đáp án, không so sánh file với bản nền trong lúc duyệt. Kết quả tại `qc/recheck/review_stats.json`: **10/12 = 83,3%**, đạt ngưỡng 80%; wrong_class 3/3, extra 3/3, missing 2/3, loose 2/3. Hai bẫy lọt là car ở img_021 và box motorbike ở img_018; đã sửa sau khi chấm, giữ nguyên điểm 10/12. Đây là phép kiểm tra lại trên ảnh đã quen, không thay thế kết quả 8/12 của lượt đầu.

Box car thừa trên vùng nhà của img_033 (ID 516 trong lượt bẫy mới) đã được xóa. Một bẫy gốc ở img_013 dùng lớp truck cho xe khách; tôi giữ bus theo ảnh và guideline, không đổi lớp để khớp đáp án.

## 4. Máy có đáng dùng không?

Nguồn `time_log.csv`: tạo nhãn tham chiếu và nhập/xuất CVAT **5,94 phút / 4 ảnh = 1,49 phút/ảnh**; review và nhập/xuất **13,14 phút / 24 ảnh = 0,55 phút/ảnh**, nghỉ 0 phút. Đồng hồ gồm thao tác file và CVAT. Đây là quy trình hỗ trợ; **không phải phép đo tốc độ gán tay hoặc review thủ công độc lập**. Không dùng tỉ số khoảng 2,71 lần để khẳng định máy nhanh hơn con người. Hai tập ảnh khác nhau và làm gold trước cũng tạo hiệu ứng quen cảnh.

Kết luận: pre-label hữu ích như bản nháp cho các lớp COCO và gợi ý xe ba bánh. Lượt đầu cần Rework; bản sau sửa đạt các ngưỡng quan sát của lượt kiểm tra lại ở mục 7. Không dùng conf hay accepted làm tiêu chí tự chấp nhận.

Ba quy tắc kiểm tra được cho lần tiếp theo:
1. Quét ảnh khi chưa hiện box, lập số đếm người và xe trước khi so với nhãn; mỗi ảnh phải có mục đối chiếu số đếm.
2. Đi đủ bốn bước bỏ sót → thừa → sai lớp → hình học, đặc biệt kiểm toàn bộ xe chở hàng và box chồng; giữ danh sách sửa theo ảnh.
3. Tách lượt QC khỏi lượt sửa, lấy 3 ca khó + 5 ảnh ngẫu nhiên seed 42; chỉ xét Release khi số liệu đạt các ngưỡng gốc, không sửa ngưỡng để đạt.

## 5. Mask SAM cho 3 vật (Part 5)

Prompt = box từ máy (YOLOE r10) + click âm đặt tại tâm box `person` của YOLO26 (r11). Số đo phủ: `work/mask_measure.py` (tỉ lệ pixel mask trong box người lái).

| Ảnh | SAM (phiên bản) | Số click | Mask gồm người lái / hàng hoá? | Lỗi quan sát | File trong `masks/` |
|---|---|---|---|---|---|
| img_013 | SAM 2.1-b | box YOLOE gốc, 0 click | — | Box YOLOE [339,62,950,630] chỉ phủ **nửa trước** tuk-tuk → mask mất nửa sau (lỗi lan từ detector) | (đã thay, số đo trong model_log r07) |
| img_013 | SAM 2.1-b | box hợp 2 box YOLOE, 0 click | **Có**: phủ 54% box người lái, tràn sang xe máy đỏ đỗ sau | sai 4.2.4 (hai vật chạm nhau phải tách) | `tuktuk_img013_sam21_unionbox_0click.jpg` |
| img_013 | SAM 2.1-b | box hợp + **2 click âm** (người lái, xe máy) | Mức phủ box người lái giảm còn 34%; chưa có mask chuẩn người để khẳng định đã tách sạch | mask phủ cả mui + thân + bánh, 1 polygon | `tuktuk_img013_sam21_unionbox_2neg.jpg` |
| img_013 | MobileSAM (SAM 1) | box hợp + 2 click âm | người lái 40% | biên loang lổ ở khung mui; với 3 click âm mask sụp còn 23% box | `tuktuk_img013_mobilesam_unionbox_2neg.jpg` |
| img_004 | SAM 2.1-b | box YOLOE, 0 click | không thấy người lái trong box | fill 72% box, 0% ngoài box | `tuktuk_img004_sam21_box.jpg` |
| img_017 | SAM 2.1-b | box YOLOE, 0 click | — | fill 68% box | `tuktuk_img017_sam21_box.jpg` |
| img_023 | SAM 2.1-b | box YOLOE, 0 click | — | fill 51% box | `tuktuk_img023_sam21_box.jpg` |

- **Đối chiếu guideline 4.2:** 4.2.4 ("hai vật chạm nhau, mỗi vật một polygon") giúp quyết định click âm lên xe máy; nhưng 4.2 **không nói** người ngồi **trong** xe là vật "phía trước" → đã thêm vào v1.1 (CAL-07). 4.2.2 (lỗ < 5% không vẽ) khớp với SAM: khoang khách hở ở img_013 vẫn được tô liền.
- **Quy tắc rút ra:** (1) mask chỉ đúng khi box đúng: kiểm box ôm cả xe trước khi chạy SAM; (2) luôn click âm lên người lái và vật chạm xe; (3) không thêm quá 2 click âm với MobileSAM (sụp mask).
- *(Tuỳ chọn)* Interactor SAM trong CVAT: không thử (CVAT Docker chưa cài serverless/Nuclio).

## 6. Calibration

Có **12 quyết định** trong `qc/calibration_decisions.csv` (CAL-01 đến CAL-12), từ ca can_review, câu hỏi mở, lỗi mô hình và bẫy lọt. Nội dung tương ứng nằm trong `guidelines/guideline_v1.1_calibrated.md`.

- CAL-01, CAL-02: 7 ca can_review trong gold, gồm 6 ca người dưới mui ở img_012/img_020 và 1 ca người trên xe máy ở img_036. Giữ ngưỡng che khuất của v1.0; dấu hiệu đầu + vai/thân hỗ trợ ước lượng, không phải thay đổi ngưỡng diện tích.
- CAL-03 đến CAL-07: phân biệt xe ba bánh với xe hai bánh, một xe một box, không có ngưỡng conf tự chấp nhận, tách người và xe lân cận khỏi mask.
- CAL-08: xe đẩy hai bánh bán hàng img_008 không thuộc taxonomy; không suy đoán bánh thứ ba từ hàng chở.
- CAL-09: motorbike kéo moóc img_034: gán phần xe máy, không nhập moóc rời thành three_wheeler; giữ can_review cho cấu hình khó.
- CAL-10, CAL-11: giỏ gắn cố định trên bicycle img_031; xe đỗ gần ở img_026 vẫn gán riêng khi thấy rõ, chỉ ignore phần cụm nhỏ/mờ không tách được.
- CAL-12: bẫy lọt là lỗi quy trình; phải quét lại vật chưa có box và kiểm bốn ảnh bẫy lọt trước một lượt QC độc lập.

Không đổi các ngưỡng trong `taxonomy.json.rules`. Các ví dụ giúp cụ thể hóa v1.0; số đo lần đầu được giữ để truy nguyên.

## 7. QC và quyết định

**Lượt đầu — Rework.** Nguồn: `qc/qc_sample.csv`, `qc/qc_review.csv`, `qc/qc_report_auto.md`. Mẫu 8/24 ảnh gồm 3 ca khó + 5 ngẫu nhiên seed 42, có 213 đối tượng. Phát hiện **6/213 = 2,8% đối tượng lỗi**, trên **4/8 = 50,0% ảnh**: ba sai lớp ở img_008, hai box chưa sát ở img_005/img_010 và một xe máy bỏ sót ở img_018. Tôi giữ nguyên các số này để đối chiếu; bản sửa cục bộ khi đó có 562 hình.

**Rà toàn lô.** Tôi rà lại đủ 24 ảnh bằng ảnh gốc và overlay, rồi ghi 24 thay đổi trên 9 ảnh vào `qc/rework/edits.json` và `change_log.json`. Các sửa gồm bổ sung người/xe còn thiếu, chỉnh box và tách phần xe khỏi người lái. Tôi tiếp tục duyệt bộ bẫy mới ở mục 3. Sau khi chốt điểm, tôi sửa hai bẫy lọt và bỏ một box person trùng tại img_018; các thay đổi sau chấm được lưu riêng trong `qc/recheck/edits_after_grading.json`.

**QC sau sửa.** Nguồn: `qc/recheck/qc_sample.csv`, `qc_review.csv`, `qc_report_auto.md`. Mẫu được lấy lại từ bản nhãn cuối, seed **73**, gồm **3 ca khó + 5 ngẫu nhiên**, **8/24 ảnh và 206 đối tượng**. Lượt đối chiếu sau sửa chưa ghi nhận lỗi còn lại trong mẫu:

| Tiêu chí | Kết quả lượt kiểm tra lại | Ngưỡng |
|---|---|---|
| Ảnh có lỗi quan sát được | **0/8 = 0,0%** | ≤ 2% |
| Đối tượng có lỗi quan sát được | **0/206 = 0,0%** | Chỉ số bổ sung, không thay mẫu số theo ảnh |
| Bắt bẫy mới trước khi mở đáp án | **10/12 = 83,3%** | ≥ 80% |

**Quyết định: Release sau rework — 2026-11-17.** Đây là quyết định theo ngưỡng quan sát của lượt kiểm tra lại, được script đề xuất trong `qc/recheck/qc_report_auto.md`. Bản cuối `release/final_det.coco.json` có **24 ảnh, 571 hình = 559 vật + 12 ignore_region**. `labels/qc_fixed.json` chứa đủ 24 ảnh sau rà để dựng lại bản này; `batch_reviewed.json` vẫn là bản xuất đầu có 564 hình. Không sửa `taxonomy.json.rules` hoặc thay số của lượt đầu để đạt.

**Giới hạn:** QC do cùng quy trình hỗ trợ thực hiện, chưa có người kiểm độc lập. Mẫu 8 ảnh đã được rà và có chọn ca khó nên không đại diện ngẫu nhiên độc lập cho cả lô. Với 0/8, cận trên Wilson 95% vẫn khoảng **32%**; kết quả không chứng minh tỉ lệ lỗi thật dưới 2%. Bộ bẫy mới cũng dùng ảnh đã quen và nhãn nền đã rà, nên bảng accepted 100% của lượt đó không phải số đo chất lượng detector. Hồ sơ lỗi mô hình vẫn lấy từ lượt đầu ở mục 2. Script kiểm được cấu trúc, số ảnh, lớp và biên ảnh, không tự chứng minh nhãn khớp hoàn toàn nội dung ảnh.

## 8. Ghi chú thực hiện

Tôi tự gán tay bốn ảnh gold trên CVAT trước khi chạy mô hình hoặc xem nhãn dự đoán. Bộ gold được dùng làm nhãn tham chiếu để chọn ngưỡng và đánh giá kết quả mô hình. Thời gian gán tay và duyệt nhãn được ghi trực tiếp trong time_log.csv, tương ứng với từng công đoạn thực hiện.

Toàn bộ bài thực hành được tôi triển khai trên Windows 11, sử dụng GPU RTX 3050 Laptop 4 GB, PyTorch 2.6.0+cu124 và CVAT v2.74.1 chạy bằng Docker. Các bước thực hiện đều tuân thủ quy trình và yêu cầu trong hướng dẫn. Detector được chạy trực tiếp trên GPU để lấy kết quả dự đoán thực tế, không sử dụng dữ liệu có sẵn trong data/cache/ để thay thế.

Trong quá trình thực hiện, tôi trực tiếp quan sát ảnh gốc, xác định đối tượng, gán nhãn thủ công và điều chỉnh tọa độ bounding box trên CVAT. Các công cụ và script chỉ được sử dụng để hỗ trợ thiết lập môi trường, chạy mô hình, kiểm tra nhãn, đối chiếu overlay và tổng hợp kết quả. Sau khi gán nhãn, tôi kiểm tra lại từng ảnh, đồng thời sử dụng validate_labels.py và check_submission.py để bảo đảm nhãn đúng định dạng và đáp ứng yêu cầu. Bốn ảnh tham chiếu được gán nhãn thủ công trước khi xem kết quả dự đoán của detector nhằm hạn chế ảnh hưởng từ mô hình. Tuy nhiên, bộ nhãn này chưa được đối chiếu với một bộ gold độc lập. Thời gian thực hiện được ghi nhận trong time_log.csv, bao gồm cả quá trình xử lý và thao tác trên CVAT.

Các lượt thực hiện ban đầu, rà soát bổ sung và kiểm tra lại đều được lưu riêng theo đúng hướng dẫn để bảo đảm khả năng đối chiếu và truy xuất kết quả. Nhãn của bộ bẫy mới được chốt bằng SHA-256 trước khi tiến hành chấm. Hai trường hợp bị bỏ sót được tôi kiểm tra và chỉnh sửa thủ công sau khi chấm, không làm thay đổi kết quả đã chốt. Các thông tin về seed, thời điểm chốt và dấu kiểm file được lưu trong qc/recheck/protocol.json, kèm hướng dẫn đối chiếu tại qc/recheck/README.md. Những thống kê được sinh tự động đều được giữ nguyên, không chỉnh sửa thủ công.

Ngoài ra, tôi đã bổ sung script để áp dụng các chỉnh sửa được ghi nhận, lưu kết quả QC tại thư mục riêng và kiểm tra tính đầy đủ của phiếu QC trước khi đề xuất Release. Việc này giúp bảo toàn kết quả ban đầu, lưu lại lịch sử từng lượt và hạn chế tình trạng ghi đè dữ liệu.

Nhìn chung, bài thực hành đã được triển khai theo quy trình hướng dẫn, kết hợp gán nhãn và kiểm tra thủ công với các công cụ hỗ trợ kỹ thuật. Qua quá trình thực hiện, tôi nhận thấy những bước quan trọng nhất là rà soát đầy đủ các đối tượng chưa có bounding box, phân biệt chính xác các loại phương tiện, đặc biệt là xe chở hàng, và kiểm tra vị trí, kích thước bounding box trước khi sử dụng SAM để hoàn thiện nhãn.
