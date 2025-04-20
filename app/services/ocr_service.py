import os
import logging
import numpy as np
import cv2
from typing import List, Dict, Any, Optional, Tuple
from pdf2image import convert_from_path
from paddleocr import PaddleOCR, PPStructure
import json
import re
from collections import defaultdict
import time

class OCRService:
    """Enhanced OCR service with improved table structure recognition and text processing."""
    
    def __init__(self, logger=None):
        """Initialize OCR service with necessary models and paths."""
        self.logger = logger or logging.getLogger(__name__)
        self.models_dir = os.path.join(os.getcwd(), 'app', 'models')
        self.debug_dir = os.path.join(os.getcwd(), 'storage', 'debug')
        os.makedirs(self.debug_dir, exist_ok=True)
        self.initialize_paddle_models()
    
    def initialize_paddle_models(self):
        """Initialize PaddleOCR models for structure analysis and text recognition."""
        try:
            self.logger.info("Initializing PaddleOCR models")
            
            # Initialize PP-StructureV2 for document layout and table analysis
            self.structure_engine = PPStructure(
                table=True, 
                ocr=True, 
                layout=True, 
                show_log=False,
                use_angle_cls=True, 
                lang="en",
                det_model_dir=os.path.join(self.models_dir, 'det'),
                rec_model_dir=os.path.join(self.models_dir, 'rec'),
                cls_model_dir=os.path.join(self.models_dir, 'cls'),
                table_model_dir=os.path.join(self.models_dir, 'table'),
                layout_model_dir=os.path.join(self.models_dir, 'layout')
            )
            
            # Initialize standard OCR engine for text detection
            self.ocr_engine = PaddleOCR(
                use_angle_cls=True, 
                lang="en", 
                show_log=False,
                det_model_dir=os.path.join(self.models_dir, 'det'),
                rec_model_dir=os.path.join(self.models_dir, 'rec'),
                cls_model_dir=os.path.join(self.models_dir, 'cls'),
                det_db_unclip_ratio=1.5  # Better for detecting merged cells
            )
            
            self.logger.info("PaddleOCR models initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing PaddleOCR models: {str(e)}")
            raise
    
    def process_pdf_and_extract_table_data(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Main pipeline for processing PDF and extracting employee schedule data.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of employee dictionaries with schedule information
        """
        try:
            self.logger.info(f"Starting to process PDF: {pdf_path}")
            
            # Step 1: Extract images from PDF
            images = self.extract_images_from_pdf(pdf_path)
            all_employees = []
            
            # Process each page
            for i, image in enumerate(images):
                page_num = i + 1
                self.logger.info(f"Processing page {page_num}/{len(images)}")
                
                # Step 2: Correct image orientation
                corrected_image = self.correct_image_orientation(image, page_num)
                
                # Step 3: Layout analysis
                layout_result = self.analyze_layout(corrected_image, page_num)
                
                # Step 4: Process tables in the layout - MENGGUNAKAN METODE YANG DITINGKATKAN
                tables_data = self.improved_process_tables(corrected_image, layout_result, page_num)
                
                # Debugging
                self.debug_table_data(tables_data, "after_extraction", page_num)
                
                # Step 5: Apply smart post-processing to fix common issues - MENGGUNAKAN METODE YANG DITINGKATKAN
                processed_data = self.improved_smart_post_processing(tables_data, page_num)
                
                all_employees.extend(processed_data)
            
            # Final validation and deduplication
            final_employees = self.validate_and_deduplicate(all_employees)
            
            self.logger.info(f"Extraction complete. Found {len(final_employees)} unique employees")
            
            # Save the complete JSON data to a file in the debug directory
            json_output_path = os.path.join(self.debug_dir, 'extracted_employees.json')
            with open(json_output_path, 'w', encoding='utf-8') as json_file:
                json.dump(final_employees, json_file, indent=2, ensure_ascii=False)
            self.logger.info(f"Saved complete JSON data to: {json_output_path}")
            
            # Convert to standardized format
            standardized_employees = []
            for emp in final_employees:
                standardized_emp = {
                    'id': emp.get('dienst-nummer', ''),
                    'name': emp.get('naam', ''),
                    'department': emp.get('org.eenheid', ''),
                    'schedule': emp.get('schedule', [])
                }
                standardized_employees.append(standardized_emp)
            
            # Log the full data structure to console
            self.logger.info("Complete employee data structure:")
            self.logger.info(json.dumps(standardized_employees, indent=2, ensure_ascii=False))
            
            # Log individual employee records as well
            self.logger.info("Individual employee records:")
            for i, employee in enumerate(standardized_employees):
                self.logger.info(f"Employee {i+1}: ID={employee['id']}, Name={employee['name']}, " +
                               f"Department={employee['department']}")
                self.logger.info(f"Schedule: {json.dumps(employee['schedule'])}")
            
            return standardized_employees
            
        except Exception as e:
            self.logger.error(f"Error processing PDF: {str(e)}")
            raise
    
    def extract_images_from_pdf(self, pdf_path: str) -> List[np.ndarray]:
        """
        Extract images from PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of images as numpy arrays
        """
        try:
            start_time = time.time()
            self.logger.info(f"[1/6] Extracting images from PDF: {pdf_path}")
            
            # Convert PDF pages to images with high DPI for better OCR
            images = convert_from_path(pdf_path, dpi=300, thread_count=4)
            np_images = [np.array(img) for img in images]
            
            # Save debug images
            for i, img in enumerate(np_images):
                debug_path = os.path.join(self.debug_dir, f"page_{i+1}_original.jpg")
                cv2.imwrite(debug_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            
            elapsed = time.time() - start_time
            self.logger.info(f"[1/6] Extracted {len(np_images)} images in {elapsed:.2f}s")
            return np_images
            
        except Exception as e:
            self.logger.error(f"[FAILED] [1/6] Error extracting images: {str(e)}")
            raise
    
    def correct_image_orientation(self, image: np.ndarray, page_number: int) -> np.ndarray:
        """
        Correct image orientation using PaddleOCR's angle classifier.
        
        Args:
            image: Input image as numpy array
            page_number: Page number for debugging
            
        Returns:
            Corrected image
        """
        try:
            start_time = time.time()
            self.logger.info(f"[2/6] Page {page_number}: Correcting image orientation")
            
            # Check if orientation is portrait or landscape
            h, w = image.shape[:2]
            is_portrait = h > w
            
            # For this specific use case, we know the schedule is typically in landscape
            # If the image is portrait, rotate it 90 degrees clockwise
            if is_portrait:
                self.logger.info(f"Page {page_number} is portrait, rotating to landscape")
                corrected = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            else:
                corrected = image
            
            # Apply further angle correction if needed using Paddle's angle classifier
            try:
                # Some versions of PaddleOCR don't expose the cls attribute directly
                # So we need to check if it exists first
                if hasattr(self.ocr_engine, 'cls'):
                    angle_result = self.ocr_engine.cls.get_rotate_crop_image(corrected)
                    if angle_result[0] != 0:  # If angle is not 0
                        self.logger.info(f"Additional rotation angle detected: {angle_result[0]}")
                        
                        # Apply additional rotation if needed
                        if angle_result[0] == 90:
                            corrected = cv2.rotate(corrected, cv2.ROTATE_90_CLOCKWISE)
                        elif angle_result[0] == 180:
                            corrected = cv2.rotate(corrected, cv2.ROTATE_180)
                        elif angle_result[0] == 270:
                            corrected = cv2.rotate(corrected, cv2.ROTATE_90_COUNTERCLOCKWISE)
                else:
                    self.logger.info("Angle classifier not available in this PaddleOCR version")
            except Exception as angle_err:
                self.logger.warning(f"Error in angle detection: {str(angle_err)}, using default orientation")
            
            # Save debug image
            debug_path = os.path.join(self.debug_dir, f"page_{page_number}_corrected.jpg")
            cv2.imwrite(debug_path, cv2.cvtColor(corrected, cv2.COLOR_RGB2BGR))
            
            elapsed = time.time() - start_time
            self.logger.info(f"[2/6] Corrected orientation in {elapsed:.2f}s")
            return corrected
            
        except Exception as e:
            self.logger.error(f"[FAILED] [2/6] Error correcting orientation: {str(e)}")
            # Return original image if correction fails
            return image
    
    def analyze_layout(self, image: np.ndarray, page_number: int) -> Dict:
        """
        Analyze document layout to identify tables, text regions, etc.
        
        Args:
            image: Input image
            page_number: Page number for debugging
            
        Returns:
            Layout analysis result
        """
        try:
            start_time = time.time()
            self.logger.info(f"[3/6] Page {page_number}: Analyzing document layout")
            
            # Enhance table borders to improve detection
            enhanced_image = self.enhance_table_borders(image)
            
            # Save the original and enhanced images for debugging
            original_path = os.path.join(self.debug_dir, f"page_{page_number}_original_for_layout.jpg")
            cv2.imwrite(original_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            
            # Use PP-StructureV2 for layout analysis
            layout_result = self.structure_engine(enhanced_image)
            
            # Create debug visualization
            debug_img = image.copy()
            
            for region in layout_result:
                region_type = region.get('type', 'unknown')
                box = region.get('bbox', [0, 0, 0, 0])
                
                # Draw bounding boxes with different colors based on type
                if region_type == 'table':
                    cv2.rectangle(debug_img, (int(box[0]), int(box[1])),
                                 (int(box[2]), int(box[3])), (0, 255, 0), 2)  # Green for tables
                elif region_type == 'text':
                    cv2.rectangle(debug_img, (int(box[0]), int(box[1])),
                                 (int(box[2]), int(box[3])), (255, 0, 0), 2)  # Red for text
                else:
                    cv2.rectangle(debug_img, (int(box[0]), int(box[1])),
                                 (int(box[2]), int(box[3])), (0, 0, 255), 2)  # Blue for others
            
            # Save layout debug image
            debug_path = os.path.join(self.debug_dir, f"page_{page_number}_layout.jpg")
            cv2.imwrite(debug_path, debug_img)
            
            # Save enhanced image
            enhanced_path = os.path.join(self.debug_dir, f"page_{page_number}_enhanced.jpg")
            cv2.imwrite(enhanced_path, enhanced_image)
            
            elapsed = time.time() - start_time
            self.logger.info(f"[3/6] Layout analysis completed in {elapsed:.2f}s. Found {len(layout_result)} regions")
            
            # Log details of each region for debugging
            for i, region in enumerate(layout_result):
                region_type = region.get('type', 'unknown')
                box = region.get('bbox', [0, 0, 0, 0])
                self.logger.info(f"Region {i+1}: Type={region_type}, Bbox={box}")
            
            return layout_result
            
        except Exception as e:
            self.logger.error(f"[FAILED] [3/6] Error analyzing layout: {str(e)}")
            return []
    
    def enhance_table_borders(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance table borders to improve table structure detection.
        
        Args:
            image: Input image
            
        Returns:
            Enhanced image
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Apply adaptive thresholding to get binary image
            binary = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 11, 2
            )
            
            # Define kernels for horizontal and vertical lines
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
            
            # Detect horizontal lines
            horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=3)
            horizontal_lines = cv2.dilate(horizontal_lines, horizontal_kernel, iterations=3)
            
            # Detect vertical lines
            vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=3)
            vertical_lines = cv2.dilate(vertical_lines, vertical_kernel, iterations=3)
            
            # Combine horizontal and vertical lines
            table_edges = cv2.add(horizontal_lines, vertical_lines)
            
            # Enhance original image with table edges
            enhanced = image.copy()
            enhanced[table_edges > 0] = (0, 0, 0)  # Mark edges in black
            
            return enhanced
            
        except Exception as e:
            self.logger.warning(f"Error enhancing table borders: {str(e)}")
            return image
    
    def improved_process_tables(self, image: np.ndarray, layout_result: List[Dict], page_number: int) -> List[Dict]:
        """
        Proses tabel dari analisis layout dengan optimasi untuk format Belanda.
        
        Args:
            image: Input image
            layout_result: Layout analysis result
            page_number: Page number for debugging
            
        Returns:
            List of extracted tables data
        """
        try:
            start_time = time.time()
            self.logger.info(f"[4/6] Page {page_number}: Processing tables with improved method")
            
            all_table_data = []
            table_count = 0
            
            # Proses setiap region dalam layout
            for region in layout_result:
                region_type = region.get('type', 'unknown')
                
                # Jika ini adalah region tabel
                if region_type == 'table':
                    table_count += 1
                    box = region.get('bbox', [0, 0, 0, 0])
                    
                    # Ekstrak region tabel
                    table_img = image[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                    
                    # Simpan gambar tabel untuk debugging
                    table_debug_path = os.path.join(self.debug_dir, f"page_{page_number}_table_{table_count}_raw.jpg")
                    cv2.imwrite(table_debug_path, cv2.cvtColor(table_img, cv2.COLOR_RGB2BGR))
                    
                    # STRATEGI BARU: Deteksi header Belanda terlebih dahulu
                    self.logger.info(f"Detecting Dutch headers for table {table_count}")
                    headers = self.detect_dutch_headers(table_img, table_count, page_number)
                    
                    table_data = []
                    
                    # Jika header berhasil dideteksi, gunakan metode ekstraksi yang ditingkatkan
                    if headers and 'columns' in headers:
                        self.logger.info(f"Using header-based extraction for table {table_count}")
                        table_data = self.extract_table_with_enhanced_relations(
                            table_img, headers, table_count, page_number
                        )
                    
                    # Jika metode ekstraksi berdasarkan header gagal, coba metode primer
                    if not table_data or len(table_data) < 2:
                        self.logger.info(f"Header-based extraction failed, trying primary methods")
                        primary_result = self.extract_table_with_structure(table_img, table_count, page_number)
                        
                        if primary_result and len(primary_result) >= 2:
                            table_data = primary_result
                        else:
                            # Coba deteksi pola warna yang ditingkatkan
                            self.logger.info(f"Primary extraction failed, trying improved color pattern")
                            improved_color_result = self.extract_table_with_improved_color_pattern(
                                table_img, table_count, page_number
                            )
                            
                            if improved_color_result and len(improved_color_result) >= 2:
                                table_data = improved_color_result
                            else:
                                # Fallback ke metode original color pattern
                                self.logger.info(f"Improved color pattern failed, trying original color pattern")
                                color_pattern_result = self.extract_table_with_color_pattern(
                                    table_img, table_count, page_number
                                )
                                
                                if color_pattern_result and len(color_pattern_result) >= 2:
                                    table_data = color_pattern_result
                                else:
                                    # Fallback ke OCR standar
                                    self.logger.info(f"All methods failed, using OCR fallback")
                                    table_data = self.extract_table_with_ocr_fallback(table_img, table_count, page_number)
                    
                    # Log the extracted table data for debugging
                    if table_data:
                        self.logger.info(f"Table {table_count} data sample (up to 3 rows):")
                        for i, row in enumerate(table_data[:min(3, len(table_data))]):
                            self.logger.info(f"  Row {i}: {json.dumps(row)}")
                        if len(table_data) > 3:
                            self.logger.info(f"  ... and {len(table_data) - 3} more rows")
                        
                        # Aplikasikan konversi header Belanda jika belum dilakukan
                        if table_data and len(table_data) > 0:
                            first_row = table_data[0]
                            has_dutch_keys = False
                            
                            # Cek apakah sudah menggunakan key Belanda
                            dutch_keys = ['naam', 'dienst-nummer', 'org.eenheid', 'maandag', 'dinsdag', 
                                         'woensdag', 'donderdag', 'vrijdag', 'zaterdag', 'zondag']
                            for key in first_row.keys():
                                if any(dutch_key in key.lower() for dutch_key in dutch_keys):
                                    has_dutch_keys = True
                                    break
                            
                            # Jika belum memiliki kunci Belanda, aplikasikan konversi
                            if not has_dutch_keys:
                                self.logger.info(f"Converting to Dutch headers...")
                                table_data = self.convert_to_dutch_headers(table_data)
                        
                        all_table_data.extend(table_data)
            
            # Jika tidak ada tabel yang ditemukan, coba OCR full-page sebagai upaya terakhir
            if not all_table_data and not table_count:
                self.logger.warning(f"No tables found on page {page_number}, applying full-page OCR")
                all_table_data = self.extract_table_with_ocr_fallback(image, 1, page_number)
            
            elapsed = time.time() - start_time
            self.logger.info(f"[4/6] Processed {table_count} tables in {elapsed:.2f}s, extracted {len(all_table_data)} rows")
            return all_table_data
            
        except Exception as e:
            self.logger.error(f"[FAILED] [4/6] Error in improved_process_tables: {str(e)}")
            # Fallback ke metode process_tables original jika terjadi error
            return self.process_tables(image, layout_result, page_number)
    
    def detect_dutch_headers(self, table_img: np.ndarray, table_idx: int, page_number: int) -> Dict:
        """
        Mendeteksi header bahasa Belanda khusus untuk jadwal karyawan.
        
        Args:
            table_img: Gambar tabel
            table_idx: Indeks tabel untuk debugging
            page_number: Nomor halaman
            
        Returns:
            Dictionary dengan header yang terdeteksi dan posisinya
        """
        try:
            self.logger.info(f"Detecting Dutch headers for table {table_idx} on page {page_number}")
            
            # Header Belanda yang diharapkan
            expected_headers = [
                {"name": "Naam", "type": "naam", "regex": r"naam"}, 
                {"name": "Dienst Nummer", "type": "dienst-nummer", "regex": r"dienst\s*n[ru]mm?er|dienst"},
                {"name": "Org. Eenheid", "type": "org.eenheid", "regex": r"org\.?\s*eenheid"},
                {"name": "Maandag", "type": "day", "day": "maandag", "regex": r"ma(?:andag)?(?:\s*\d{1,2}[-/\.]\d{1,2})?"},
                {"name": "Dinsdag", "type": "day", "day": "dinsdag", "regex": r"di(?:nsdag)?(?:\s*\d{1,2}[-/\.]\d{1,2})?"},
                {"name": "Woensdag", "type": "day", "day": "woensdag", "regex": r"wo(?:ensdag)?(?:\s*\d{1,2}[-/\.]\d{1,2})?"},
                {"name": "Donderdag", "type": "day", "day": "donderdag", "regex": r"do(?:nderdag)?(?:\s*\d{1,2}[-/\.]\d{1,2})?"},
                {"name": "Vrijdag", "type": "day", "day": "vrijdag", "regex": r"vr(?:ijdag)?(?:\s*\d{1,2}[-/\.]\d{1,2})?"},
                {"name": "Zaterdag", "type": "day", "day": "zaterdag", "regex": r"za(?:terdag)?(?:\s*\d{1,2}[-/\.]\d{1,2})?"},
                {"name": "Zondag", "type": "day", "day": "zondag", "regex": r"zo(?:ndag)?(?:\s*\d{1,2}[-/\.]\d{1,2})?"}
            ]
            
            # Deteksi header row menggunakan OCR pada 15% bagian atas gambar
            header_region_height = int(table_img.shape[0] * 0.15)
            header_region = table_img[:header_region_height, :]
            
            # Simpan gambar region header untuk debugging
            header_debug_path = os.path.join(self.debug_dir, f"page_{page_number}_table_{table_idx}_header_region.jpg")
            cv2.imwrite(header_debug_path, header_region)
            
            # OCR pada region header
            ocr_result = self.ocr_engine.ocr(header_region, cls=True)
            
            if not ocr_result or not ocr_result[0]:
                self.logger.warning(f"No OCR results found in header region")
                return {}
            
            # Ekstrak dan analisis teks header
            header_items = []
            for line in ocr_result[0]:
                if len(line) >= 2:
                    bbox = line[0]
                    text = line[1][0]
                    confidence = line[1][1]
                    
                    # Skip yang kosong atau confidence rendah
                    if not text.strip() or confidence < 0.5:
                        continue
                    
                    # Hitung koordinat bounding box
                    points = np.array(bbox)
                    x_min = min(points[:, 0])
                    x_max = max(points[:, 0])
                    y_min = min(points[:, 1])
                    y_max = max(points[:, 1])
                    center_x = np.mean(points[:, 0])
                    center_y = np.mean(points[:, 1])
                    
                    header_items.append({
                        'text': text,
                        'bbox': bbox,
                        'x_min': x_min,
                        'x_max': x_max,
                        'y_min': y_min,
                        'y_max': y_max,
                        'center_x': center_x,
                        'center_y': center_y,
                        'width': x_max - x_min,
                        'height': y_max - y_min,
                        'confidence': confidence
                    })
            
            # Kelompokkan header berdasarkan baris (clustering y-coordinate)
            eps = header_region_height * 0.1  # 10% dari tinggi region header sebagai threshold
            clustered_headers = self.simple_row_clustering(header_items, eps)
            
            # Group berdasarkan cluster baris
            rows_by_cluster = defaultdict(list)
            for item in clustered_headers:
                rows_by_cluster[item['row_cluster']].append(item)
            
            # Urutkan cluster berdasarkan koordinat y (atas ke bawah)
            sorted_clusters = sorted(
                rows_by_cluster.keys(), 
                key=lambda c: min(item['center_y'] for item in rows_by_cluster[c])
            )
            
            # Cluster pertama dengan minimal 3 item kemungkinan besar adalah baris header
            header_row = []
            for cluster in sorted_clusters:
                items = rows_by_cluster[cluster]
                if len(items) >= 3:  # Minimal 3 item header untuk dianggap baris header
                    # Urutkan berdasarkan koordinat x (kiri ke kanan)
                    header_row = sorted(items, key=lambda x: x['center_x'])
                    break
            
            if not header_row:
                self.logger.warning("Could not identify a clear header row")
                return {}
            
            # Buat visualisasi header untuk debugging
            header_viz = header_region.copy()
            for item in header_row:
                pts = np.array(item['bbox'], np.int32)
                cv2.polylines(header_viz, [pts], True, (0, 255, 0), 2)
                cv2.putText(header_viz, item['text'], 
                          (int(item['x_min']), int(item['y_min'])-5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            header_viz_path = os.path.join(self.debug_dir, f"page_{page_number}_table_{table_idx}_headers_detected.jpg")
            cv2.imwrite(header_viz_path, header_viz)
            
            # Match teks header dengan header yang diharapkan
            headers = {}
            column_mapping = {}
            
            for i, item in enumerate(header_row):
                text = item['text'].lower().strip()
                matched = False
                
                for expected in expected_headers:
                    regex = expected["regex"]
                    if re.search(regex, text, re.IGNORECASE):
                        # Ekstrak tanggal jika ada untuk header hari
                        date = ""
                        if expected["type"] == "day":
                            date_match = re.search(r'(\d{1,2}[-/\.]\d{1,2}(?:[-/\.]\d{2,4})?)', text)
                            if date_match:
                                date = date_match.group(1)
                        
                        column_mapping[i] = {
                            'type': expected["type"],
                            'name': expected["name"],
                            'original_text': item['text'],
                            'x_center': item['center_x'],
                            'x_min': item['x_min'],
                            'x_max': item['x_max'],
                            'y_min': item['y_min'],
                            'y_max': item['y_max']
                        }
                        
                        # Tambahkan hari dan tanggal untuk kolom hari
                        if expected["type"] == "day":
                            column_mapping[i]["day"] = expected["day"]
                            column_mapping[i]["date"] = date
                            
                        matched = True
                        break
                
                # Jika tidak ada match, tambahkan sebagai kolom tidak diketahui
                if not matched:
                    column_mapping[i] = {
                        'type': 'unknown',
                        'name': item['text'],
                        'original_text': item['text'],
                        'x_center': item['center_x'],
                        'x_min': item['x_min'],
                        'x_max': item['x_max'],
                        'y_min': item['y_min'],
                        'y_max': item['y_max']
                    }
            
            # Hitung batas kolom untuk ekstraksi data
            if column_mapping:
                # Urutkan berdasarkan indeks kolom
                sorted_columns = [column_mapping[i] for i in sorted(column_mapping.keys())]
                
                # Definisikan batas kolom berdasarkan posisi header
                for i in range(len(sorted_columns) - 1):
                    current = sorted_columns[i]
                    next_col = sorted_columns[i + 1]
                    
                    # Set batas antara kolom saat ini dan berikutnya
                    right_boundary = (current['x_max'] + next_col['x_min']) / 2
                    current['right_boundary'] = right_boundary
                    
                    if i == 0:
                        # Untuk kolom pertama, set batas kiri berdasarkan posisi
                        current['left_boundary'] = max(0, current['x_min'] - 10)
                    else:
                        # Untuk kolom lain, batas kiri adalah batas kanan kolom sebelumnya
                        current['left_boundary'] = sorted_columns[i-1]['right_boundary']
                
                # Set batas untuk kolom terakhir
                last_col = sorted_columns[-1]
                last_col['left_boundary'] = sorted_columns[-2]['right_boundary'] if len(sorted_columns) > 1 else 0
                last_col['right_boundary'] = table_img.shape[1]  # Tepi kanan gambar
                
                # Set batas baris header untuk ekstraksi data
                header_y_max = max(col['y_max'] for col in sorted_columns)
                
                headers = {
                    'columns': sorted_columns,
                    'header_bottom': header_y_max,
                    'original_mapping': column_mapping
                }
            
            return headers
            
        except Exception as e:
            self.logger.error(f"Error detecting Dutch headers: {str(e)}")
            return {}
    
    def extract_table_with_enhanced_relations(self, table_img: np.ndarray, headers: Dict, table_idx: int, page_number: int) -> List[Dict]:
        """
        Ekstrak data tabel dengan relasi baris-kolom yang ditingkatkan menggunakan header yang terdeteksi.
        
        Args:
            table_img: Gambar tabel
            headers: Informasi header yang terdeteksi
            table_idx: Indeks tabel untuk debugging
            page_number: Nomor halaman untuk debugging
            
        Returns:
            List of rows with cell data mapped to appropriate headers
        """
        try:
            self.logger.info(f"Extracting table with enhanced row-column relationships")
            
            if not headers or 'columns' not in headers:
                self.logger.warning("No valid headers provided for enhanced extraction")
                return []
            
            columns = headers['columns']
            header_bottom = headers['header_bottom']
            
            # Ekstrak tubuh tabel (area di bawah header)
            table_body = table_img[int(header_bottom):, :]
            
            # Simpan gambar badan tabel untuk debugging
            body_debug_path = os.path.join(self.debug_dir, f"page_{page_number}_table_{table_idx}_body.jpg")
            cv2.imwrite(body_debug_path, table_body)
            
            # STEP 1: Deteksi baris dengan warna bergantian
            self.logger.info("Detecting alternating color rows")
            rows = self.improved_detect_alternating_color_rows(table_body, table_idx, page_number)
            
            if not rows:
                self.logger.warning("Failed to detect alternating color rows, trying standard detection")
                rows = self.detect_alternating_color_rows(table_body, table_idx, page_number)
                
                if not rows:
                    self.logger.warning("All row detection methods failed")
                    return []
            
            # Buat visualisasi baris
            row_viz = table_body.copy()
            for i, row in enumerate(rows):
                color = (0, 255, 0) if row.get('is_gray') else (0, 0, 255)
                cv2.rectangle(row_viz, (0, row['top']), (row_viz.shape[1], row['bottom']), color, 2)
                cv2.putText(row_viz, f"Row {i}", (10, row['top'] + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Simpan visualisasi baris
            row_viz_path = os.path.join(self.debug_dir, f"page_{page_number}_table_{table_idx}_detected_rows.jpg")
            cv2.imwrite(row_viz_path, row_viz)
            
            # STEP 2: Proses setiap baris dan selaraskan dengan kolom
            all_rows_data = []
            
            # Tambahkan baris header
            header_row = {}
            for col in columns:
                col_type = col['type']
                if col_type == 'day':
                    key = col['day']
                    # Add date if available
                    if col.get('date'):
                        header_row[key] = f"{col['name']} {col['date']}"
                    else:
                        header_row[key] = col['name']
                else:
                    header_row[col_type] = col['name']
            
            all_rows_data.append(header_row)
            
            # Proses data rows
            for row_idx, row in enumerate(rows):
                row_img = table_body[row['top']:row['bottom'], :]
                
                # OCR pada seluruh baris
                ocr_result = self.ocr_engine.ocr(row_img, cls=True)
                row_data = {}
                
                if ocr_result and ocr_result[0]:
                    # Ekstrak semua elemen teks dalam baris
                    text_elements = []
                    for line in ocr_result[0]:
                        if len(line) >= 2:
                            bbox = line[0]
                            text = line[1][0]
                            confidence = line[1][1]
                            
                            # Skip yang kosong atau confidence rendah
                            if not text.strip() or confidence < 0.5:
                                continue
                            
                            # Hitung koordinat bounding box
                            points = np.array(bbox)
                            x_min = min(points[:, 0])
                            x_max = max(points[:, 0])
                            y_min = min(points[:, 1])
                            y_max = max(points[:, 1])
                            center_x = np.mean(points[:, 0])
                            
                            text_elements.append({
                                'text': text,
                                'x_min': x_min,
                                'x_max': x_max,
                                'y_min': y_min,
                                'y_max': y_max,
                                'center_x': center_x
                            })
                    
                    # Map elemen teks ke kolom berdasarkan koordinat x
                    for col in columns:
                        col_elements = []
                        for elem in text_elements:
                            # Periksa apakah pusat elemen berada dalam batas kolom
                            if col['left_boundary'] <= elem['center_x'] <= col['right_boundary']:
                                col_elements.append(elem)
                        
                        # Urutkan elemen berdasarkan koordinat x
                        col_elements.sort(key=lambda e: e['center_x'])
                        
                        # Gabungkan semua teks dalam kolom ini
                        if col_elements:
                            combined_text = " ".join([elem['text'] for elem in col_elements])
                            
                            # Map ke kunci yang sesuai berdasarkan tipe kolom
                            if col['type'] == 'day':
                                # Untuk kolom hari, gunakan nama hari
                                row_data[col['day']] = combined_text
                            else:
                                # Untuk kolom lain, gunakan tipe kolom
                                row_data[col['type']] = combined_text
                
                # Jika kita menemukan data, tambahkan barisnya
                if row_data:
                    # Isi nilai kosong untuk kolom yang hilang
                    for col in columns:
                        if col['type'] == 'day' and col['day'] not in row_data:
                            row_data[col['day']] = ""
                        elif col['type'] not in row_data:
                            row_data[col['type']] = ""
                    
                    all_rows_data.append(row_data)
            
            # Buat visualisasi dengan batas kolom
            col_viz = table_img.copy()
            for col in columns:
                # Gambar batas kolom
                cv2.line(col_viz, 
                       (int(col['left_boundary']), 0), 
                       (int(col['left_boundary']), col_viz.shape[0]), 
                       (0, 255, 0), 1)
                cv2.line(col_viz, 
                       (int(col['right_boundary']), 0), 
                       (int(col['right_boundary']), col_viz.shape[0]), 
                       (0, 0, 255), 1)
                
                # Label kolom
                cv2.putText(col_viz, col['name'], 
                          (int((col['left_boundary'] + col['right_boundary']) / 2 - 40), 20),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            # Simpan visualisasi kolom
            col_viz_path = os.path.join(self.debug_dir, f"page_{page_number}_table_{table_idx}_columns.jpg")
            cv2.imwrite(col_viz_path, col_viz)
            
            self.logger.info(f"Enhanced extraction complete, found {len(all_rows_data)} rows")
            return all_rows_data
            
        except Exception as e:
            self.logger.error(f"Error in enhanced table extraction: {str(e)}")
            return []
    
    def extract_table_with_improved_color_pattern(self, table_img: np.ndarray, table_idx: int, page_number: int) -> List[Dict]:
        """
        Ekstrak tabel menggunakan deteksi baris warna bergantian yang ditingkatkan.
        
        Args:
            table_img: Gambar tabel
            table_idx: Indeks tabel
            page_number: Nomor halaman untuk debugging
            
        Returns:
            List of rows with cell data
        """
        try:
            self.logger.info(f"Using improved color pattern detection for table {table_idx} on page {page_number}")
            
            # Deteksi header untuk digunakan sebagai referensi
            header_region_height = int(table_img.shape[0] * 0.15)
            header_region = table_img[:header_region_height, :]
            headers = self.detect_dutch_headers(table_img, table_idx, page_number)
            
            # Deteksi baris berdasarkan warna bergantian
            rows = self.improved_detect_alternating_color_rows(table_img, table_idx, page_number)
            self.logger.info(f"Detected {len(rows)} rows based on improved color pattern")
            
            if not rows:
                # Fallback ke metode original
                rows = self.detect_alternating_color_rows(table_img, table_idx, page_number)
                self.logger.info(f"Fallback method detected {len(rows)} rows")
                
                if not rows:
                    return []
            
            # Proses setiap baris dengan OCR
            all_data = []
            
            # Ekstrak baris header (baris pertama)
            if rows:
                header_row = rows[0]
                header_img = table_img[header_row['top']:header_row['bottom'], :]
                header_ocr = self.ocr_engine.ocr(header_img, cls=True)
                
                # Buat data header
                header_data = {}
                
                if headers and 'columns' in headers:
                    # Gunakan informasi header yang terdeteksi
                    for col in headers['columns']:
                        if col['type'] == 'day':
                            header_data[col['day']] = f"{col['name']} {col.get('date', '')}"
                        else:
                            header_data[col['type']] = col['name']
                else:
                    # Generik header dari OCR header row jika headers tidak terdeteksi
                    if header_ocr and header_ocr[0]:
                        texts_with_positions = []
                        for line in header_ocr[0]:
                            if len(line) >= 2:
                                bbox = line[0]
                                text = line[1][0]
                                confidence = line[1][1]
                                
                                if not text.strip() or confidence < 0.5:
                                    continue
                                    
                                points = np.array(bbox)
                                center_x = np.mean(points[:, 0])
                                texts_with_positions.append((text, center_x))
                        
                        # Urutkan berdasarkan posisi x dan buat kolom
                        texts_with_positions.sort(key=lambda x: x[1])
                        
                        # Coba match dengan header Belanda yang umum
                        for i, (text, _) in enumerate(texts_with_positions):
                            text_lower = text.lower()
                            
                            # Deteksi tipe header
                            if 'naam' in text_lower:
                                header_data['naam'] = text
                            elif 'dienst' in text_lower or 'nummer' in text_lower:
                                header_data['dienst-nummer'] = text
                            elif 'org' in text_lower or 'eenheid' in text_lower:
                                header_data['org.eenheid'] = text
                            elif 'ma' in text_lower or 'maandag' in text_lower:
                                header_data['maandag'] = text
                            elif 'di' in text_lower or 'dinsdag' in text_lower:
                                header_data['dinsdag'] = text
                            elif 'wo' in text_lower or 'woensdag' in text_lower:
                                header_data['woensdag'] = text
                            elif 'do' in text_lower or 'donderdag' in text_lower:
                                header_data['donderdag'] = text
                            elif 'vr' in text_lower or 'vrijdag' in text_lower:
                                header_data['vrijdag'] = text
                            elif 'za' in text_lower or 'zaterdag' in text_lower:
                                header_data['zaterdag'] = text
                            elif 'zo' in text_lower or 'zondag' in text_lower:
                                header_data['zondag'] = text
                            else:
                                header_data[f"col_{i}"] = text
                
                if header_data:
                    all_data.append(header_data)
                
                # Proses baris data
                for i, row in enumerate(rows[1:], 1):  # Skip header row
                    row_img = table_img[row['top']:row['bottom'], :]
                    row_ocr = self.ocr_engine.ocr(row_img, cls=True)
                    
                    row_data = {}
                    
                    if row_ocr and row_ocr[0]:
                        texts_with_positions = []
                        for line in row_ocr[0]:
                            if len(line) >= 2:
                                bbox = line[0]
                                text = line[1][0]
                                confidence = line[1][1]
                                
                                if not text.strip() or confidence < 0.5:
                                    continue
                                    
                                points = np.array(bbox)
                                center_x = np.mean(points[:, 0])
                                texts_with_positions.append((text, center_x))
                        
                        # Jika kita memiliki header dengan info kolom
                        if headers and 'columns' in headers:
                            for col in headers['columns']:
                                col_elements = []
                                for text, x_pos in texts_with_positions:
                                    if col['left_boundary'] <= x_pos <= col['right_boundary']:
                                        col_elements.append(text)
                                
                                if col_elements:
                                    # Gabungkan teks untuk kolom ini
                                    combined_text = " ".join(col_elements)
                                    
                                    # Tentukan key berdasarkan tipe kolom
                                    if col['type'] == 'day':
                                        key = col['day']
                                    else:
                                        key = col['type']
                                    
                                    row_data[key] = combined_text
                        else:
                            # Tanpa info kolom, coba sesuaikan dengan header
                            # Urutkan berdasarkan posisi x
                            texts_with_positions.sort(key=lambda x: x[1])
                            
                            # Dapatkan header keys secara berurutan
                            header_keys = list(header_data.keys())
                            
                            # Tetapkan teks ke kolom yang sesuai
                            for j, (text, _) in enumerate(texts_with_positions):
                                if j < len(header_keys):
                                    row_data[header_keys[j]] = text
                                else:
                                    row_data[f"col_{j}"] = text
                    
                    if row_data:
                        # Isi nilai kosong untuk kolom yang hilang
                        for key in header_data.keys():
                            if key not in row_data:
                                row_data[key] = ""
                        
                        all_data.append(row_data)
            
            return all_data
            
        except Exception as e:
            self.logger.error(f"Error in improved color pattern extraction: {str(e)}")
            return []
    
    def improved_detect_alternating_color_rows(self, table_img: np.ndarray, table_idx: int, page_number: int) -> List[Dict]:
        """
        Deteksi baris dengan warna bergantian yang ditingkatkan (fokus untuk template Belanda).
        
        Args:
            table_img: Gambar tabel
            table_idx: Indeks tabel
            page_number: Nomor halaman
            
        Returns:
            List of dictionaries dengan batas baris
        """
        try:
            # Langkah 1: Coba deteksi header terlebih dahulu dengan OCR pada region atas
            header_height = int(table_img.shape[0] * 0.15)  # 15% bagian atas untuk header
            header_region = table_img[:header_height, :]
            
            # Simpan gambar region header untuk debugging
            debug_path = os.path.join(self.debug_dir, f"page_{page_number}_table_{table_idx}_header_region.jpg")
            cv2.imwrite(debug_path, header_region)
            
            # Langkah 2: Gunakan HSV untuk deteksi pola warna (lebih sensitif terhadap variasi abu-abu)
            hsv = cv2.cvtColor(table_img, cv2.COLOR_BGR2HSV)
            
            # Definisikan range warna abu-abu (saturation rendah, value medium)
            # Kunci: Gunakan rentang yang lebih luas untuk menangkap variasi halus
            lower_gray = np.array([0, 0, 150])
            upper_gray = np.array([180, 30, 230])
            
            # Buat mask untuk pixel abu-abu
            gray_mask = cv2.inRange(hsv, lower_gray, upper_gray)
            
            # Simpan mask untuk debugging
            mask_path = os.path.join(self.debug_dir, f"page_{page_number}_table_{table_idx}_gray_mask_improved.jpg")
            cv2.imwrite(mask_path, gray_mask)
            
            # Langkah 3: Analisis profil vertikal dengan smoothing untuk mengurangi noise
            row_profile_raw = np.sum(gray_mask, axis=1) / gray_mask.shape[1]
            
            # Smoothing menggunakan moving average untuk mengurangi noise
            kernel_size = 5
            kernel = np.ones(kernel_size) / kernel_size
            row_profile = np.convolve(row_profile_raw, kernel, mode='same')
            
            # Langkah 4: Thresholding adaptif berdasarkan histogram distribusi nilai
            # Mencoba menemukan threshold yang lebih optimal
            # Gunakan metode Otsu jika tersedia, atau fallback ke nilai threshold tengah
            threshold = 128  # Nilai default
            try:
                # Konversi profile ke format numpy yang tepat untuk Otsu
                profile_for_otsu = (row_profile * 255 / np.max(row_profile)).astype(np.uint8)
                # Reshape untuk Otsu
                profile_for_otsu = profile_for_otsu.reshape(-1, 1)
                
                if cv2.__version__ >= '3.0.0':
                    _, threshold_value = cv2.threshold(profile_for_otsu, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                    threshold = threshold_value / 255.0 * np.max(row_profile)
                    self.logger.info(f"Using Otsu threshold: {threshold}")
            except Exception as e:
                self.logger.warning(f"Otsu thresholding failed: {str(e)}, using default threshold")
            
            is_gray = row_profile > threshold
            
            # Langkah 5: Identifikasi transisi dengan lebih robust
            transitions = []
            for i in range(1, len(is_gray)):
                if is_gray[i] != is_gray[i-1]:
                    transitions.append(i)
            
            # Langkah 6: Buat baris dari transisi dengan validasi lebar minimal
            rows = []
            i = 0
            
            # Jika tidak ada transisi, mungkin semua baris berwarna sama
            # Coba deteksi baris dengan metode alternatif
            if len(transitions) < 2:
                self.logger.warning(f"Kurang dari 2 transisi terdeteksi, coba metode alternatif")
                # Estimasi tinggi baris berdasarkan ukuran font umum
                est_row_height = int(table_img.shape[0] / 25)  # Perkiraan 25 baris per halaman
                est_rows = int(table_img.shape[0] / est_row_height)
                
                for r in range(est_rows):
                    top = r * est_row_height
                    bottom = min((r+1) * est_row_height, table_img.shape[0])
                    
                    rows.append({
                        'top': top,
                        'bottom': bottom,
                        'is_gray': r % 2 == 1  # Asumsikan baris ganjil adalah abu-abu
                    })
            else:
                # Proses transisi normal
                while i < len(transitions) - 1:
                    top = transitions[i]
                    bottom = transitions[i+1]
                    
                    # Hanya pertimbangkan baris dengan tinggi minimal
                    if bottom - top > 10:
                        rows.append({
                            'top': top,
                            'bottom': bottom,
                            'is_gray': is_gray[top]
                        })
                    i += 2
            
            # Langkah 7: Pastikan kita memiliki baris header (dibagian atas tabel)
            if not rows or rows[0]['top'] > 20:  # Jika baris pertama tidak dekat dengan atas
                # Tambahkan baris header manual
                header_height = 30
                if rows:
                    avg_height = sum(row['bottom'] - row['top'] for row in rows) / len(rows)
                    header_height = max(25, int(avg_height * 0.8))  # Sedikit lebih pendek dari rata-rata
                
                header_row = {
                    'top': 0,
                    'bottom': header_height,
                    'is_gray': False  # Header biasanya berlatar putih
                }
                
                rows.insert(0, header_row)
            
            # Langkah 8: Visualisasi untuk debugging
            vis_img = table_img.copy()
            for i, row in enumerate(rows):
                color = (0, 255, 0) if row['is_gray'] else (0, 0, 255)
                cv2.rectangle(vis_img, (0, row['top']), (vis_img.shape[1], row['bottom']), color, 2)
                cv2.putText(vis_img, f"Row {i}", (10, row['top'] + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Simpan visualisasi
            vis_path = os.path.join(self.debug_dir, f"page_{page_number}_table_{table_idx}_color_rows_improved.jpg")
            cv2.imwrite(vis_path, vis_img)
            
            self.logger.info(f"Terdeteksi {len(rows)} baris menggunakan metode yang ditingkatkan")
            return rows
            
        except Exception as e:
            self.logger.error(f"Error pada improved_detect_alternating_color_rows: {str(e)}")
            # Fallback ke metode original jika terjadi error
            return self.detect_alternating_color_rows(table_img, table_idx, page_number)
    
    def convert_to_dutch_headers(self, table_data: List[Dict]) -> List[Dict]:
        """
        Konversi header standar ke format Belanda.
        
        Args:
            table_data: Data tabel dengan header generic
            
        Returns:
            Data tabel dengan header Belanda
        """
        if not table_data or len(table_data) < 1:
            return table_data
        
        # Dapatkan header asli
        header = table_data[0]
        
        # Mapping header
        dutch_header_map = {
            'name': 'naam',
            'id': 'dienst-nummer',
            'department': 'org.eenheid',
            'monday': 'maandag',
            'tuesday': 'dinsdag',
            'wednesday': 'woensdag',
            'thursday': 'donderdag',
            'friday': 'vrijdag',
            'saturday': 'zaterdag',
            'sunday': 'zondag'
        }
        
        # Mapping header posisional untuk col_X
        positional_map = {
            'col_0': 'naam',
            'col_1': 'dienst-nummer',
            'col_2': 'org.eenheid',
            'col_3': 'maandag',
            'col_4': 'dinsdag',
            'col_5': 'woensdag',
            'col_6': 'donderdag',
            'col_7': 'vrijdag',
            'col_8': 'zaterdag',
            'col_9': 'zondag'
        }
        
        # Konversi header
        new_header = {}
        for key, value in header.items():
            # Cek jika key ada dalam mapping
            if key.lower() in dutch_header_map:
                new_header[dutch_header_map[key.lower()]] = value
            # Cek jika key adalah col_X
            elif key in positional_map:
                new_header[positional_map[key]] = value
            # Text matching untuk header Belanda
            elif 'naam' in key.lower():
                new_header['naam'] = value
            elif 'dienst' in key.lower() or 'nummer' in key.lower():
                new_header['dienst-nummer'] = value
            elif 'org' in key.lower() or 'eenheid' in key.lower():
                new_header['org.eenheid'] = value
            elif 'ma' in key.lower() or 'maandag' in key.lower():
                new_header['maandag'] = value
            elif 'di' in key.lower() or 'dinsdag' in key.lower():
                new_header['dinsdag'] = value
            elif 'wo' in key.lower() or 'woensdag' in key.lower():
                new_header['woensdag'] = value
            elif 'do' in key.lower() or 'donderdag' in key.lower():
                new_header['donderdag'] = value
            elif 'vr' in key.lower() or 'vrijdag' in key.lower():
                new_header['vrijdag'] = value
            elif 'za' in key.lower() or 'zaterdag' in key.lower():
                new_header['zaterdag'] = value
            elif 'zo' in key.lower() or 'zondag' in key.lower():
                new_header['zondag'] = value
            else:
                new_header[key] = value
        
        # Konversi data rows
        new_data = [new_header]
        
        for row in table_data[1:]:
            new_row = {}
            for key, value in row.items():
                # Untuk setiap key di row, cari key yang sesuai di header baru
                mapped_key = None
                
                if key.lower() in dutch_header_map:
                    mapped_key = dutch_header_map[key.lower()]
                elif key in positional_map:
                    mapped_key = positional_map[key]
                elif 'naam' in key.lower():
                    mapped_key = 'naam'
                elif 'dienst' in key.lower() or 'nummer' in key.lower():
                    mapped_key = 'dienst-nummer'
                elif 'org' in key.lower() or 'eenheid' in key.lower():
                    mapped_key = 'org.eenheid'
                elif 'ma' in key.lower() or 'maandag' in key.lower():
                    mapped_key = 'maandag'
                elif 'di' in key.lower() or 'dinsdag' in key.lower():
                    mapped_key = 'dinsdag'
                elif 'wo' in key.lower() or 'woensdag' in key.lower():
                    mapped_key = 'woensdag'
                elif 'do' in key.lower() or 'donderdag' in key.lower():
                    mapped_key = 'donderdag'
                elif 'vr' in key.lower() or 'vrijdag' in key.lower():
                    mapped_key = 'vrijdag'
                elif 'za' in key.lower() or 'zaterdag' in key.lower():
                    mapped_key = 'zaterdag'
                elif 'zo' in key.lower() or 'zondag' in key.lower():
                    mapped_key = 'zondag'
                else:
                    mapped_key = key
                
                new_row[mapped_key] = value
            
            new_data.append(new_row)
        
        return new_data
    
    def improved_smart_post_processing(self, rows_data: List[Dict], page_number: int) -> List[Dict]:
        """
        Versi yang ditingkatkan dari smart_post_processing yang lebih baik menangani format Belanda.
        
        Args:
            rows_data: List of extracted rows
            page_number: Page number for debugging
            
        Returns:
            List of employee records with corrected data
        """
        try:
            start_time = time.time()
            self.logger.info(f"[5/6] Page {page_number}: Improved smart post-processing of {len(rows_data)} rows")
            
            # Jika tidak ada data, return empty list
            if not rows_data:
                self.logger.warning(f"No rows data to process on page {page_number}")
                return []
            
            # Log sample data untuk debugging
            for i, row in enumerate(rows_data[:min(3, len(rows_data))]):
                self.logger.info(f"Row {i} sample: {json.dumps(row)}")
            
            # Periksa struktur data untuk debugging
            has_appropriate_keys = False
            if rows_data and len(rows_data) > 0:
                first_row = rows_data[0]
                # Cek apakah ada kunci yang sesuai dengan header Belanda
                dutch_keys = ['naam', 'dienst-nummer', 'org.eenheid',
                              'maandag', 'dinsdag', 'woensdag', 'donderdag', 'vrijdag', 'zaterdag', 'zondag']
                
                for key in first_row.keys():
                    if any(dutch_key in key.lower() for dutch_key in dutch_keys):
                        has_appropriate_keys = True
                        break
                
                self.logger.info(f"Data has appropriate Dutch keys: {has_appropriate_keys}")
                
                # Jika masih menggunakan kunci generik (col_X), coba konversi
                if not has_appropriate_keys:
                    self.logger.info("Converting generic column keys to Dutch headers")
                    rows_data = self.convert_to_dutch_headers(rows_data)
            
            # Sekarang gunakan improved_infer_column_types untuk mendapatkan deteksi kolom yang lebih baik
            if rows_data and len(rows_data) > 0:
                column_types = self.improved_infer_column_types(rows_data[0])
                self.logger.info(f"Improved inferred column types: {column_types}")
                
                # Step 2: Process rows to employee records with improved extraction
                employee_records = []
                for i, row in enumerate(rows_data):
                    # Skip header row
                    if i == 0:
                        continue
                    
                    # Extract employee information with improved method
                    employee = self.improved_extract_employee_from_row(row, column_types)
                    
                    if employee:
                        employee_records.append(employee)
                
                # Step 3: Use original post-processing for the remainder
                corrected_records = []
                
                # First pass: flag potential fragments
                for i, record in enumerate(employee_records):
                    # Check if name ends with comma (potential fragment)
                    if record['naam'] and record['naam'].strip().endswith(','):
                        record['is_name_fragment'] = True
                    else:
                        record['is_name_fragment'] = False
                    
                    # Flag potential number fragments (names that are just numbers)
                    if record['naam'] and re.match(r'^\d+$', record['naam'].strip()):
                        record['is_nummer_fragment'] = True
                    else:
                        record['is_nummer_fragment'] = False
                    
                    # Flag potential schedule fragments (names that look like times)
                    if record['naam'] and re.search(r'\d{1,2}:\d{2}', record['naam']):
                        record['is_schedule_fragment'] = True
                    else:
                        record['is_schedule_fragment'] = False
                
                # Second pass: merge fragments and fix misclassifications
                skip_indices = set()
                for i, record in enumerate(employee_records):
                    if i in skip_indices:
                        continue
                    
                    corrected_record = record.copy()
                    
                    # Fix name fragments
                    if record.get('is_name_fragment') and i + 1 < len(employee_records):
                        next_record = employee_records[i + 1]
                        
                        # Check if the next record looks like initials or surname (2-4 uppercase letters)
                        if re.match(r'^[A-Z]{2,4}', next_record.get('naam', '').strip()):
                            corrected_record['naam'] = f"{record['naam']} {next_record['naam']}"
                            skip_indices.add(i + 1)
                    
                    # Fix number fragments (names that are just numbers)
                    if record.get('is_nummer_fragment'):
                        corrected_record['dienst-nummer'] = record['naam']
                        
                        # If there's a previous record with name and no number, assign this number to it
                        if i > 0 and not employee_records[i-1]['dienst-nummer'] and employee_records[i-1]['naam']:
                            corrected_records[-1]['dienst-nummer'] = record['naam']
                            continue  # Skip adding this as a separate record
                    
                    # Fix schedule fragments (names that look like times)
                    if record.get('is_schedule_fragment'):
                        # Don't add this as a separate employee record
                        if i > 0 and employee_records[i-1]['naam']:
                            # If the previous record already has a schedule, try to add to it
                            if corrected_records and corrected_records[-1]['schedule']:
                                # Add to the last day's schedule
                                if corrected_records[-1]['schedule']:
                                    last_day = corrected_records[-1]['schedule'][-1]
                                    parsed_entries = self.improved_parse_schedule_entry(record['naam'])
                                    if parsed_entries:
                                        last_day['schedule'].extend(parsed_entries)
                            continue
                    
                    # Remove processing flags
                    for flag in ['is_name_fragment', 'is_nummer_fragment', 'is_schedule_fragment']:
                        if flag in corrected_record:
                            del corrected_record[flag]
                    
                    # Handle mixed number and name cases (extract number from name if necessary)
                    if ' ' in corrected_record['naam'] and any(part.isdigit() and len(part) > 5 for part in corrected_record['naam'].split()):
                        parts = corrected_record['naam'].split()
                        for part in parts:
                            if part.isdigit() and len(part) > 5:
                                # This is likely a dienst-nummer
                                corrected_record['dienst-nummer'] = part
                                # Remove ID from name
                                corrected_record['naam'] = ' '.join([p for p in parts if p != part])
                    
                    # Clean up the record data
                    corrected_record['naam'] = corrected_record['naam'].strip()
                    corrected_record['dienst-nummer'] = corrected_record['dienst-nummer'].strip()
                    corrected_record['org.eenheid'] = corrected_record['org.eenheid'].strip()
                    
                    # Validate and add to final records
                    if self.is_valid_employee_record(corrected_record):
                        corrected_records.append(corrected_record)
                
                elapsed = time.time() - start_time
                self.logger.info(f"[5/6] Improved post-processing complete in {elapsed:.2f}s. Found {len(corrected_records)} valid records")
                return corrected_records
            else:
                # Fallback to original method
                self.logger.warning(f"Invalid or empty rows_data, falling back to original method")
                return self.smart_post_processing(rows_data, page_number)
            
        except Exception as e:
            self.logger.error(f"Error in improved_smart_post_processing: {str(e)}")
            # Fallback to original method
            return self.smart_post_processing(rows_data, page_number)
    
    def improved_infer_column_types(self, headers: Dict) -> Dict:
        """
        Infer column types dengan pengenalan header Belanda yang lebih baik.
        
        Args:
            headers: Row data dengan headers
            
        Returns:
            Dictionary mapping column names to types with additional date info
        """
        # Pertama gunakan fungsi yang sudah ada untuk mendapatkan hasil awal
        column_types = self.infer_column_types(headers)
        
        # Jika column_types kosong atau tidak lengkap, coba deteksi dengan pola Belanda yang lebih spesifik
        if not column_types or len(column_types) < 4:  # Minimal seharusnya ada nama, id, departemen, dan setidaknya 1 hari
            self.logger.info("Menjalankan deteksi header Belanda yang ditingkatkan")
            
            # Header Belanda yang lebih spesifik
            specific_dutch_patterns = {
                'naam': [r'naam'],
                'dienst-nummer': [r'dienst\s*n[ru]mm?er', r'^dienst$', r'^nr\.?$', r'^nummer$'],
                'org.eenheid': [r'org\.?\s*eenheid', r'^org', r'^eenheid$', r'^afdeling$'],
                'maandag': [r'^ma(?:andag)?$', r'ma(?:andag)?\s+\d{1,2}[-/\.]\d{1,2}'],
                'dinsdag': [r'^di(?:nsdag)?$', r'di(?:nsdag)?\s+\d{1,2}[-/\.]\d{1,2}'],
                'woensdag': [r'^wo(?:ensdag)?$', r'wo(?:ensdag)?\s+\d{1,2}[-/\.]\d{1,2}'],
                'donderdag': [r'^do(?:nderdag)?$', r'do(?:nderdag)?\s+\d{1,2}[-/\.]\d{1,2}'],
                'vrijdag': [r'^vr(?:ijdag)?$', r'vr(?:ijdag)?\s+\d{1,2}[-/\.]\d{1,2}'],
                'zaterdag': [r'^za(?:terdag)?$', r'za(?:terdag)?\s+\d{1,2}[-/\.]\d{1,2}'],
                'zondag': [r'^zo(?:ndag)?$', r'zo(?:ndag)?\s+\d{1,2}[-/\.]\d{1,2}']
            }
            
            # Coba match dengan pola yang lebih spesifik
            new_column_types = {}
            for col, value in headers.items():
                value_text = str(value).lower().strip()
                matched = False
                
                for key, patterns in specific_dutch_patterns.items():
                    for pattern in patterns:
                        if re.search(pattern, value_text, re.IGNORECASE):
                            # Ekstrak tanggal jika ada dan ini adalah kolom hari
                            if key in ['maandag', 'dinsdag', 'woensdag', 'donderdag', 'vrijdag', 'zaterdag', 'zondag']:
                                date_match = re.search(r'(\d{1,2}[-/\.]\d{1,2}(?:[-/\.]\d{2,4})?)', value_text)
                                date = date_match.group(1) if date_match else ""
                                
                                new_column_types[col] = {
                                    'type': 'day',
                                    'day': key,
                                    'date': date
                                }
                            else:
                                new_column_types[col] = {'type': key}
                            
                            matched = True
                            break
                    
                    if matched:
                        break
            
            # Jika deteksi baru menemukan lebih banyak kolom, gunakan itu
            if len(new_column_types) > len(column_types):
                self.logger.info(f"Deteksi yang ditingkatkan menemukan {len(new_column_types)} kolom vs {len(column_types)} original")
                return new_column_types
            else:
                # Jika keduanya tidak menemukan apa-apa, coba mode darurat
                if not column_types and not new_column_types:
                    # Coba dengan posisi kolom
                    emergency_column_types = {}
                    cols = list(headers.keys())
                    
                    # Prioritas: jika ada 'naam' atau 'name' di value, gunakan itu
                    for col, value in headers.items():
                        value_text = str(value).lower().strip()
                        if 'naam' in value_text or 'name' in value_text:
                            emergency_column_types[col] = {'type': 'naam'}
                        elif 'dienst' in value_text or 'nummer' in value_text:
                            emergency_column_types[col] = {'type': 'dienst-nummer'}
                        elif 'org' in value_text or 'eenheid' in value_text:
                            emergency_column_types[col] = {'type': 'org.eenheid'}
                    
                    # Jika masih tidak ada, gunakan posisi
                    if not emergency_column_types:
                        if cols:
                            col_types = ['naam', 'dienst-nummer', 'org.eenheid']
                            for i, col in enumerate(cols[:min(3, len(cols))]):
                                emergency_column_types[col] = {'type': col_types[i]}
                            
                            # Tambahkan hari-hari
                            days = ['maandag', 'dinsdag', 'woensdag', 'donderdag', 'vrijdag', 'zaterdag', 'zondag']
                            for i, col in enumerate(cols[3:min(10, len(cols))]):
                                if i < len(days):
                                    emergency_column_types[col] = {'type': 'day', 'day': days[i], 'date': ''}
                    
                    if emergency_column_types:
                        self.logger.info(f"Emergency column detection found {len(emergency_column_types)} columns")
                        return emergency_column_types
        
        return column_types
    
    def improved_extract_employee_from_row(self, row: Dict, column_types: Dict) -> Dict:
        """
        Ekstrak informasi karyawan dari baris dengan deteksi yang ditingkatkan.
        
        Args:
            row: Data baris
            column_types: Dictionary mapping column names to types
            
        Returns:
            Employee dictionary dalam format yang diminta
        """
        employee = {
            'naam': '',
            'dienst-nummer': '',
            'org.eenheid': '',
            'schedule': []
        }
        
        # Cari kolom yang cocok dengan tipe yang diharapkan
        for column, value in row.items():
            column_lower = column.lower()
            
            # Skip nilai kosong
            if not value:
                continue
            
            # Cek pencocokan langsung dari column_types
            if column in column_types:
                column_info = column_types[column]
                column_type = column_info.get('type', '')
                
                # Basic fields
                if column_type == 'naam':
                    employee['naam'] = str(value).strip()
                elif column_type == 'dienst-nummer':
                    employee['dienst-nummer'] = str(value).strip()
                elif column_type == 'org.eenheid':
                    employee['org.eenheid'] = str(value).strip()
                # Extract schedule for days
                elif column_type == 'day':
                    day = column_info.get('day', '')
                    date = column_info.get('date', '')
                    schedule_value = str(value).strip()
                    
                    if schedule_value and schedule_value.lower() != 'off':
                        # Parse the schedule text with improved method
                        schedule_entries = self.improved_parse_schedule_entry(schedule_value)
                        
                        # Add to employee schedule
                        if schedule_entries:
                            employee['schedule'].append({
                                'day': day,
                                'date': date,
                                'schedule': schedule_entries
                            })
            else:
                # Cek secara langsung berdasarkan nama kolom
                if 'naam' in column_lower or column_lower == 'name':
                    employee['naam'] = str(value).strip()
                elif 'dienst' in column_lower or 'nummer' in column_lower or column_lower == 'id':
                    employee['dienst-nummer'] = str(value).strip()
                elif 'org' in column_lower or 'eenheid' in column_lower or column_lower == 'department':
                    employee['org.eenheid'] = str(value).strip()
                # Cek untuk nama hari
                elif 'maandag' in column_lower or column_lower == 'monday':
                    schedule_value = str(value).strip()
                    if schedule_value and schedule_value.lower() != 'off':
                        schedule_entries = self.improved_parse_schedule_entry(schedule_value)
                        if schedule_entries:
                            employee['schedule'].append({
                                'day': 'maandag',
                                'date': '',
                                'schedule': schedule_entries
                            })
                elif 'dinsdag' in column_lower or column_lower == 'tuesday':
                    schedule_value = str(value).strip()
                    if schedule_value and schedule_value.lower() != 'off':
                        schedule_entries = self.improved_parse_schedule_entry(schedule_value)
                        if schedule_entries:
                            employee['schedule'].append({
                                'day': 'dinsdag',
                                'date': '',
                                'schedule': schedule_entries
                            })
                elif 'woensdag' in column_lower or column_lower == 'wednesday':
                    schedule_value = str(value).strip()
                    if schedule_value and schedule_value.lower() != 'off':
                        schedule_entries = self.improved_parse_schedule_entry(schedule_value)
                        if schedule_entries:
                            employee['schedule'].append({
                                'day': 'woensdag',
                                'date': '',
                                'schedule': schedule_entries
                            })
                elif 'donderdag' in column_lower or column_lower == 'thursday':
                    schedule_value = str(value).strip()
                    if schedule_value and schedule_value.lower() != 'off':
                        schedule_entries = self.improved_parse_schedule_entry(schedule_value)
                        if schedule_entries:
                            employee['schedule'].append({
                                'day': 'donderdag',
                                'date': '',
                                'schedule': schedule_entries
                            })
                elif 'vrijdag' in column_lower or column_lower == 'friday':
                    schedule_value = str(value).strip()
                    if schedule_value and schedule_value.lower() != 'off':
                        schedule_entries = self.improved_parse_schedule_entry(schedule_value)
                        if schedule_entries:
                            employee['schedule'].append({
                                'day': 'vrijdag',
                                'date': '',
                                'schedule': schedule_entries
                            })
                elif 'zaterdag' in column_lower or column_lower == 'saturday':
                    schedule_value = str(value).strip()
                    if schedule_value and schedule_value.lower() != 'off':
                        schedule_entries = self.improved_parse_schedule_entry(schedule_value)
                        if schedule_entries:
                            employee['schedule'].append({
                                'day': 'zaterdag',
                                'date': '',
                                'schedule': schedule_entries
                            })
                elif 'zondag' in column_lower or column_lower == 'sunday':
                    schedule_value = str(value).strip()
                    if schedule_value and schedule_value.lower() != 'off':
                        schedule_entries = self.improved_parse_schedule_entry(schedule_value)
                        if schedule_entries:
                            employee['schedule'].append({
                                'day': 'zondag',
                                'date': '',
                                'schedule': schedule_entries
                            })
        
        return employee
    
    def improved_parse_schedule_entry(self, schedule_text: str) -> List[str]:
        """
        Parse jadwal dengan dukungan yang lebih baik untuk format Belanda.
        
        Args:
            schedule_text: Teks jadwal
            
        Returns:
            List of schedule entries
        """
        # Langkah 1: Coba menggunakan parser yang ada
        original_entries = self.parse_schedule_entry(schedule_text)
        
        # Jika parser yang sudah ada berhasil menemukan entri, gunakan itu
        if original_entries:
            return original_entries
        
        # Langkah 2: Coba metode parsing yang lebih kuat jika yang asli gagal
        text = schedule_text.strip()
        if not text or text.lower() == 'off':
            return []
        
        entries = []
        
        # Format yang sering muncul dalam jadwal Belanda:
        # 1. DIENST 07:00 16:30
        # 2. [Rust] 13:00 22:30
        # 3. [Vr.zondag] 00:00 24:00
        # 4. Kadang shift ditulis tanpa spasi: DIENST07:0016:30
        
        # Pola regex yang lebih kuat untuk menangkap waktu
        time_pattern = r'(\d{1,2}[:\.]\d{2})'
        service_pattern = r'(DIENST|RUST|\[.*?\])'
        
        # Pola untuk format standar: DIENST 07:00 16:30
        std_pattern = re.compile(f"{service_pattern}\\s*{time_pattern}\\s*{time_pattern}", re.IGNORECASE)
        std_match = std_pattern.search(text)
        if std_match:
            service = std_match.group(1)
            start_time = std_match.group(2)
            end_time = std_match.group(3)
            entries.append(f"{service} {start_time} {end_time}")
            return entries
        
        # Pola untuk hanya waktu: 07:00 16:30
        times_pattern = re.compile(f"{time_pattern}\\s*{time_pattern}")
        times_match = times_pattern.search(text)
        if times_match:
            start_time = times_match.group(1)
            end_time = times_match.group(2)
            entries.append(f"DIENST {start_time} {end_time}")  # Default ke DIENST
            return entries
        
        # Pola untuk format tanpa spasi: DIENST07:0016:30
        compact_pattern = re.compile(f"{service_pattern}?({time_pattern})({time_pattern})", re.IGNORECASE)
        compact_match = compact_pattern.search(text)
        if compact_match:
            service = compact_match.group(1) if compact_match.group(1) else "DIENST"
            start_time = compact_match.group(2)
            end_time = compact_match.group(3)
            entries.append(f"{service} {start_time} {end_time}")
            return entries
        
        # Jika masih belum ada yang cocok, cari semua waktu di string
        all_times = re.findall(time_pattern, text)
        if len(all_times) >= 2:
            # Asumsikan dua waktu pertama adalah awal dan akhir shift
            entries.append(f"DIENST {all_times[0]} {all_times[1]}")
            return entries
        
        # Jika tidak ada pola yang cocok, tambahkan teks asli
        if not entries:
            entries.append(text)
        
        return entries
    
    def debug_table_data(self, table_data: List[Dict], label: str, page_number: int) -> None:
        """
        Helper untuk debugging table data.
        
        Args:
            table_data: Data tabel untuk di-debug
            label: Label untuk mengidentifikasi tahap debugging
            page_number: Nomor halaman
        """
        if not table_data:
            self.logger.warning(f"[DEBUG][{label}] No table data on page {page_number}")
            return
        
        self.logger.info(f"[DEBUG][{label}] Table data on page {page_number} has {len(table_data)} rows")
        
        # Header analysis
        if len(table_data) > 0:
            header = table_data[0]
            self.logger.info(f"[DEBUG][{label}] Header keys: {list(header.keys())}")
            
            # Check for specific keys we expect
            expected_keys = ['naam', 'dienst-nummer', 'org.eenheid', 'maandag', 'dinsdag', 'woensdag', 
                             'donderdag', 'vrijdag', 'zaterdag', 'zondag']
            found_keys = [key for key in expected_keys if any(key.lower() in k.lower() for k in header.keys())]
            self.logger.info(f"[DEBUG][{label}] Found expected keys: {found_keys}")
            
            # Check for generic column keys
            generic_keys = [k for k in header.keys() if k.startswith('col_')]
            if generic_keys:
                self.logger.info(f"[DEBUG][{label}] Found generic keys: {generic_keys}")
        
        # Data row analysis
        if len(table_data) > 1:
            # Sample some data rows
            for i, row in enumerate(table_data[1:min(4, len(table_data))]):
                self.logger.info(f"[DEBUG][{label}] Data row {i+1}: {json.dumps(row)}")
                
                # Check for empty values
                empty_keys = [k for k, v in row.items() if not v or v.strip() == '']
                if empty_keys:
                    self.logger.info(f"[DEBUG][{label}] Empty values in row {i+1}: {empty_keys}")
        
        # Save full data for detailed inspection
        debug_path = os.path.join(self.debug_dir, f"page_{page_number}_{label}_table_data.json")
        with open(debug_path, 'w', encoding='utf-8') as f:
            json.dump(table_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"[DEBUG][{label}] Saved detailed data to {debug_path}")
    
    def extract_table_with_structure(self, table_img: np.ndarray, table_idx: int, page_number: int) -> List[Dict]:
        """
        Extract table using PP-StructureV2's table recognition.
        
        Args:
            table_img: Table image
            table_idx: Table index
            page_number: Page number for debugging
            
        Returns:
            List of rows with cell data
        """
        try:
            self.logger.info(f"Using PP-StructureV2 for table {table_idx} on page {page_number}")
            
            # Use PPStructure to recognize table structure
            result = self.structure_engine(table_img)
            
            # Find table object in result
            table_data = []
            for item in result:
                if item.get('type') == 'table':
                    if 'res' in item and isinstance(item['res'], dict) and 'html' in item['res']:
                        # Save HTML for debugging
                        html_path = os.path.join(self.debug_dir, f"page_{page_number}_table_{table_idx}.html")
                        with open(html_path, 'w', encoding='utf-8') as f:
                            f.write(item['res']['html'])
                        
                        # Parse rows from structured data
                        raw_rows = item['res'].get('cells', [])
                        headers = []
                        header_row_idx = -1
                        
                        # Identify header row (usually the first non-empty row)
                        for i, row in enumerate(raw_rows):
                            if row and any(cell.get('text') for cell in row):
                                headers = [cell.get('text', '').strip() for cell in row]
                                header_row_idx = i
                                break
                        
                        # Create header row for data - IMPROVED: Try to detect Dutch headers
                        header_data = {}
                        dutch_header_detected = False
                        
                        # Try to match Dutch headers
                        dutch_headers = {
                            'naam': ['naam'],
                            'dienst-nummer': ['dienst', 'nummer', 'nr'],
                            'org.eenheid': ['org', 'eenheid', 'afdeling'],
                            'maandag': ['maandag', 'ma'],
                            'dinsdag': ['dinsdag', 'di'],
                            'woensdag': ['woensdag', 'wo'],
                            'donderdag': ['donderdag', 'do'],
                            'vrijdag': ['vrijdag', 'vr'],
                            'zaterdag': ['zaterdag', 'za'],
                            'zondag': ['zondag', 'zo']
                        }
                        
                        # Check for Dutch header patterns
                        for i, header in enumerate(headers):
                            header_lower = header.lower().strip()
                            matched = False
                            
                            for dutch_key, patterns in dutch_headers.items():
                                if any(pattern in header_lower for pattern in patterns):
                                    header_data[dutch_key] = header
                                    dutch_header_detected = True
                                    matched = True
                                    break
                            
                            # If no match, use generic column name
                            if not matched:
                                header_data[f"col_{i}"] = header
                        
                        # If no Dutch headers detected, use generic col_X approach
                        if not dutch_header_detected:
                            header_data = {}
                            for i, header in enumerate(headers):
                                header_data[f"col_{i}"] = header
                        
                        if header_data:
                            table_data.append(header_data)
                        
                        # Process data rows (rows after header)
                        for i, row in enumerate(raw_rows):
                            if i <= header_row_idx:
                                continue
                            
                            if not row or not any(cell.get('text') for cell in row):
                                continue
                            
                            # Create row data
                            row_data = {}
                            
                            if dutch_header_detected:
                                # Map to Dutch headers if detected
                                dutch_header_map = {}
                                for j, header in enumerate(headers):
                                    header_lower = header.lower().strip()
                                    for dutch_key, patterns in dutch_headers.items():
                                        if any(pattern in header_lower for pattern in patterns):
                                            dutch_header_map[j] = dutch_key
                                            break
                                
                                # Use mapping to assign values
                                for j, cell in enumerate(row):
                                    if j in dutch_header_map:
                                        key = dutch_header_map[j]
                                    else:
                                        key = f"col_{j}"
                                    
                                    value = cell.get('text', '').strip()
                                    row_data[key] = value
                            else:
                                # Use generic col_X keys
                                for j, cell in enumerate(row):
                                    col_key = f"col_{j}"
                                    value = cell.get('text', '').strip()
                                    row_data[col_key] = value
                            
                            if row_data:
                                table_data.append(row_data)
                        
                        # Add semantic recovery and relationship detection
                        semantic_table_data = self.dutch_semantic_entity_recognition(table_data)
                        self.logger.info(f"Semantic recognition applied to {len(table_data)} rows")
                        
                        # Use semantic recovery if it produced results, otherwise keep original
                        if semantic_table_data and len(semantic_table_data) >= len(table_data):
                            table_data = semantic_table_data
            
            self.logger.info(f"PP-StructureV2 extracted {len(table_data)} rows from table {table_idx}")
            return table_data
            
        except Exception as e:
            self.logger.warning(f"Error extracting table with PP-StructureV2: {str(e)}")
            return []
    
    def dutch_semantic_entity_recognition(self, table_data: List[Dict]) -> List[Dict]:
        """
        Apply semantic entity recognition specifically for Dutch employee records.
        
        Args:
            table_data: Raw table data
            
        Returns:
            Semantically enhanced table data with Dutch field names
        """
        try:
            if not table_data or len(table_data) < 2:  # Need at least header and one data row
                return table_data
            
            # Try to identify column semantics from header
            header = table_data[0]
            column_semantics = {}
            
            # Keywords for matching different column types in Dutch
            name_keywords = ['naam', 'name', 'employee', 'medewerker', 'person']
            id_keywords = ['dienst nummer', 'dienst-nummer', 'dienstnummer', 'dienst nr', 'dienst', 'nummer', 'code', 'no.', 'no']
            dept_keywords = ['org.eenheid', 'org eenheid', 'orgeenheid', 'org', 'eenheid', 'afdeling', 'department', 'dept', 'unit', 'team']
            day_keywords = {
                'maandag': ['maandag', 'ma', 'monday', 'mon'],
                'dinsdag': ['dinsdag', 'di', 'tuesday', 'tue'],
                'woensdag': ['woensdag', 'wo', 'wednesday', 'wed'],
                'donderdag': ['donderdag', 'do', 'thursday', 'thu'],
                'vrijdag': ['vrijdag', 'vr', 'friday', 'fri'],
                'zaterdag': ['zaterdag', 'za', 'saturday', 'sat'],
                'zondag': ['zondag', 'zo', 'sunday', 'sun']
            }
            
            # Identify column semantics
            for col, value in header.items():
                if not value:
                    continue
                
                value_lower = value.lower()
                
                # Check for name column
                if any(keyword in value_lower for keyword in name_keywords):
                    column_semantics[col] = 'naam'
                    continue
                
                # Check for ID column
                if any(keyword in value_lower for keyword in id_keywords):
                    column_semantics[col] = 'dienst-nummer'
                    continue
                
                # Check for department column
                if any(keyword in value_lower for keyword in dept_keywords):
                    column_semantics[col] = 'org.eenheid'
                    continue
                
                # Check for day columns with dynamic dates
                for day, keywords in day_keywords.items():
                    # First check for simple match (e.g., "Maandag")
                    if any(keyword in value_lower for keyword in keywords):
                        column_semantics[col] = day
                        break
                    
                    # Then check for day with date (e.g., "Maandag 12-05")
                    # Regular expression to match day name followed by date
                    for keyword in keywords:
                        if re.search(rf"{keyword}\s+\d{{1,2}}[-/\.]\d{{1,2}}(?:[-/\.]\d{{2,4}})?", value_lower):
                            column_semantics[col] = day
                            break
            
            # If no semantics detected, try to infer from position and column naming
            if not column_semantics:
                for col in header.keys():
                    # Check for column names that might indicate specific columns
                    col_lower = col.lower()
                    
                    if 'naam' in col_lower:
                        column_semantics[col] = 'naam'
                    elif 'dienst' in col_lower or 'nummer' in col_lower:
                        column_semantics[col] = 'dienst-nummer'
                    elif 'org' in col_lower or 'eenheid' in col_lower:
                        column_semantics[col] = 'org.eenheid'
                    elif col.startswith('ma') or 'maandag' in col_lower:
                        column_semantics[col] = 'maandag'
                    elif col.startswith('di') or 'dinsdag' in col_lower:
                        column_semantics[col] = 'dinsdag'
                    elif col.startswith('wo') or 'woensdag' in col_lower:
                        column_semantics[col] = 'woensdag'
                    elif col.startswith('do') or 'donderdag' in col_lower:
                        column_semantics[col] = 'donderdag'
                    elif col.startswith('vr') or 'vrijdag' in col_lower:
                        column_semantics[col] = 'vrijdag'
                    elif col.startswith('za') or 'zaterdag' in col_lower:
                        column_semantics[col] = 'zaterdag'
                    elif col.startswith('zo') or 'zondag' in col_lower:
                        column_semantics[col] = 'zondag'
            
            # If still no semantics detected, use positional inference
            if not column_semantics:
                columns = list(header.keys())
                if len(columns) >= 1:
                    column_semantics[columns[0]] = 'naam'
                if len(columns) >= 2:
                    column_semantics[columns[1]] = 'dienst-nummer'
                if len(columns) >= 3:
                    column_semantics[columns[2]] = 'org.eenheid'
                
                # Assign remaining columns as days of week
                days = ['maandag', 'dinsdag', 'woensdag', 'donderdag', 'vrijdag', 'zaterdag', 'zondag']
                for i, col in enumerate(columns[3:min(10, len(columns))]):
                    if i < len(days):
                        column_semantics[col] = days[i]
            
            # Apply semantics to rows
            enhanced_rows = []
            
            # Create a new header with semantic keys
            new_header = {}
            for col, semantic_key in column_semantics.items():
                new_header[semantic_key] = header[col]
            
            # Add the new header
            enhanced_rows.append(new_header)
            
            # Process data rows
            for row in table_data[1:]:  # Skip header
                enhanced_row = {}
                
                # Map columns to semantic meanings
                for col, value in row.items():
                    if col in column_semantics:
                        semantic_key = column_semantics[col]
                        enhanced_row[semantic_key] = value
                    else:
                        # Keep original column if no semantic meaning
                        enhanced_row[col] = value
                
                enhanced_rows.append(enhanced_row)
            
            return enhanced_rows
            
        except Exception as e:
            self.logger.warning(f"Error in Dutch semantic entity recognition: {str(e)}")
            return table_data
    
    def extract_table_with_color_pattern(self, table_img: np.ndarray, table_idx: int, page_number: int) -> List[Dict]:
        """
        Extract table by detecting alternating colored rows.
        
        Args:
            table_img: Table image
            table_idx: Table index
            page_number: Page number for debugging
            
        Returns:
            List of rows with cell data
        """
        try:
            self.logger.info(f"Using color pattern detection for table {table_idx} on page {page_number}")
            
            # Detect rows based on alternating colors
            rows = self.detect_alternating_color_rows(table_img, table_idx, page_number)
            self.logger.info(f"Detected {len(rows)} rows based on color pattern")
            
            if not rows:
                return []
            
            # Process each row with OCR
            all_data = []
            
            # Extract header row (first row)
            if rows:
                header_row = rows[0]
                header_img = table_img[header_row['top']:header_row['bottom'], :]
                header_ocr = self.ocr_engine.ocr(header_img, cls=True)
                
                # Create header data
                header_data = {}
                
                if header_ocr and header_ocr[0]:
                    texts_with_positions = []
                    for line in header_ocr[0]:
                        if len(line) >= 2:
                            bbox = line[0]
                            text = line[1][0]
                            confidence = line[1][1]
                            
                            if not text.strip() or confidence < 0.5:
                                continue
                                
                            points = np.array(bbox)
                            center_x = np.mean(points[:, 0])
                            texts_with_positions.append((text, center_x))
                    
                    # Sort by x position and create columns
                    texts_with_positions.sort(key=lambda x: x[1])
                    
                    # Improved: Try to match Dutch headers
                    dutch_headers = {
                        'naam': ['naam'],
                        'dienst-nummer': ['dienst', 'nummer', 'nr'],
                        'org.eenheid': ['org', 'eenheid', 'afdeling'],
                        'maandag': ['maandag', 'ma'],
                        'dinsdag': ['dinsdag', 'di'],
                        'woensdag': ['woensdag', 'wo'],
                        'donderdag': ['donderdag', 'do'],
                        'vrijdag': ['vrijdag', 'vr'],
                        'zaterdag': ['zaterdag', 'za'],
                        'zondag': ['zondag', 'zo']
                    }
                    
                    for i, (text, _) in enumerate(texts_with_positions):
                        text_lower = text.lower()
                        matched = False
                        
                        for dutch_key, patterns in dutch_headers.items():
                            if any(pattern in text_lower for pattern in patterns):
                                header_data[dutch_key] = text
                                matched = True
                                break
                        
                        if not matched:
                            header_data[f"col_{i}"] = text
                
                if header_data:
                    all_data.append(header_data)
                
                # Process data rows
                for i, row in enumerate(rows[1:], 1):  # Skip header row
                    row_img = table_img[row['top']:row['bottom'], :]
                    row_ocr = self.ocr_engine.ocr(row_img, cls=True)
                    
                    row_data = {}
                    
                    if row_ocr and row_ocr[0]:
                        texts_with_positions = []
                        for line in row_ocr[0]:
                            if len(line) >= 2:
                                bbox = line[0]
                                text = line[1][0]
                                confidence = line[1][1]
                                
                                if not text.strip() or confidence < 0.5:
                                    continue
                                    
                                points = np.array(bbox)
                                center_x = np.mean(points[:, 0])
                                texts_with_positions.append((text, center_x))
                        
                        # Sort by x position and create cells
                        texts_with_positions.sort(key=lambda x: x[1])
                        
                        # Match with header position
                        header_positions = []
                        for key, value in header_data.items():
                            # Find position of this header in the original texts_with_positions
                            for j, (h_text, h_pos) in enumerate(texts_with_positions):
                                if h_text == value:
                                    header_positions.append((key, h_pos))
                                    break
                        
                        # If we have header positions, use them to align cells
                        if header_positions:
                            for text, pos in texts_with_positions:
                                # Find closest header
                                closest_header = min(header_positions, key=lambda h: abs(h[1] - pos))
                                header_key = closest_header[0]
                                
                                # Append text to this column
                                if header_key in row_data:
                                    row_data[header_key] += " " + text
                                else:
                                    row_data[header_key] = text
                        else:
                            # Fallback to simple ordering
                            for j, (text, _) in enumerate(texts_with_positions):
                                col_key = f"col_{j}"
                                if col_key in header_data:
                                    # If we have this column in header, use the Dutch key if available
                                    dutch_col = next((k for k, v in header_data.items() 
                                                  if k != col_key and v == header_data[col_key]), col_key)
                                    row_data[dutch_col] = text
                                else:
                                    row_data[col_key] = text
                    
                    if row_data:
                        all_data.append(row_data)
            
            # Apply semantic entity recognition to ensure proper mapping
            if all_data and len(all_data) > 1:
                enhanced_data = self.dutch_semantic_entity_recognition(all_data)
                if enhanced_data and len(enhanced_data) >= len(all_data):
                    return enhanced_data
            
            return all_data
            
        except Exception as e:
            self.logger.error(f"Error in color pattern extraction: {str(e)}")
            return []
    
    def detect_alternating_color_rows(self, table_img: np.ndarray, table_idx: int, page_number: int) -> List[Dict]:
        """
        Detect rows in a table based on alternating background colors (gray and white).
        
        Args:
            table_img: Table image
            table_idx: Table index for debugging
            page_number: Page number for debugging
            
        Returns:
            List of dictionaries with row boundaries
        """
        try:
            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(table_img, cv2.COLOR_BGR2HSV)
            
            # Define range for gray color (low saturation, medium value)
            lower_gray = np.array([0, 0, 160])
            upper_gray = np.array([180, 25, 220])
            
            # Create mask for gray pixels
            gray_mask = cv2.inRange(hsv, lower_gray, upper_gray)
            
            # Save the mask for debugging
            mask_path = os.path.join(self.debug_dir, f"page_{page_number}_table_{table_idx}_gray_mask.jpg")
            cv2.imwrite(mask_path, gray_mask)
            
            # Sum pixels horizontally to get vertical profile
            # This gives us a histogram showing rows with gray backgrounds
            row_profile = np.sum(gray_mask, axis=1) / gray_mask.shape[1]
            
            # Threshold to identify gray regions (rows)
            threshold = 128  # Midpoint between 0 and 255
            is_gray = row_profile > threshold
            
            # Find transitions between gray and non-gray regions
            transitions = []
            for i in range(1, len(is_gray)):
                if is_gray[i] != is_gray[i-1]:
                    transitions.append(i)
            
            # Group transitions to form rows
            rows = []
            i = 0
            while i < len(transitions) - 1:
                top = transitions[i]
                bottom = transitions[i+1]
                
                # Only consider rows with minimum height
                if bottom - top > 10:
                    rows.append({
                        'top': top,
                        'bottom': bottom,
                        'is_gray': is_gray[top]
                    })
                i += 2
            
            # Create visualization
            vis_img = cv2.cvtColor(table_img.copy(), cv2.COLOR_BGR2RGB)
            for i, row in enumerate(rows):
                color = (0, 255, 0) if row['is_gray'] else (255, 0, 0)
                cv2.rectangle(vis_img, (0, row['top']), (vis_img.shape[1], row['bottom']), color, 2)
                cv2.putText(vis_img, f"Row {i}", (10, row['top'] + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Save visualization
            vis_path = os.path.join(self.debug_dir, f"page_{page_number}_table_{table_idx}_color_rows.jpg")
            cv2.imwrite(vis_path, vis_img)
            
            return rows
            
        except Exception as e:
            self.logger.error(f"Error detecting alternating color rows: {str(e)}")
            return []
    
    def extract_table_with_ocr_fallback(self, table_img: np.ndarray, table_idx: int, page_number: int) -> List[Dict]:
        """
        Extract table using more adaptive OCR-based approach as fallback.
        
        Args:
            table_img: Table image
            table_idx: Table index
            page_number: Page number for debugging
            
        Returns:
            List of rows with cell data
        """
        try:
            self.logger.info(f"Using OCR fallback for table {table_idx} on page {page_number}")
            
            # Perform OCR on the table image
            ocr_result = self.ocr_engine.ocr(table_img, cls=True)
            
            if not ocr_result or not ocr_result[0]:
                self.logger.warning(f"No OCR results found for table {table_idx}")
                return []
            
            # Extract text and coordinates
            text_items = []
            for line in ocr_result[0]:
                if len(line) >= 2:
                    bbox = line[0]
                    text = line[1][0]
                    confidence = line[1][1]
                    
                    # Skip empty or low-confidence items
                    if not text.strip() or confidence < 0.5:
                        continue
                    
                    # Calculate bounding box coordinates
                    points = np.array(bbox)
                    x_min = min(points[:, 0])
                    x_max = max(points[:, 0])
                    y_min = min(points[:, 1])
                    y_max = max(points[:, 1])
                    center_x = np.mean(points[:, 0])
                    center_y = np.mean(points[:, 1])
                    
                    text_items.append({
                        'text': text,
                        'bbox': bbox,
                        'x_min': x_min,
                        'x_max': x_max,
                        'y_min': y_min,
                        'y_max': y_max,
                        'center_x': center_x,
                        'center_y': center_y,
                        'width': x_max - x_min,
                        'height': y_max - y_min,
                        'confidence': confidence
                    })
            
            # Group text items into rows using simple clustering on y-coordinates
            if not text_items:
                return []
            
            # Simple row clustering with adaptive threshold based on image height
            eps = table_img.shape[0] * 0.02  # 2% of image height - adaptive threshold
            clustered_items = self.simple_row_clustering(text_items, eps)
            
            # Group by row cluster
            rows_by_cluster = defaultdict(list)
            for item in clustered_items:
                rows_by_cluster[item['row_cluster']].append(item)
            
            # Sort text within each row by x-coordinate (left to right)
            for cluster in rows_by_cluster:
                rows_by_cluster[cluster].sort(key=lambda x: x['center_x'])
            
            # Create debug visualization
            debug_img = cv2.cvtColor(table_img, cv2.COLOR_RGB2BGR)
            colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), 
                     (255, 255, 0), (0, 255, 255), (255, 0, 255),
                     (128, 0, 0), (0, 128, 0), (0, 0, 128)]
            
            for cluster, items in rows_by_cluster.items():
                color = colors[cluster % len(colors)]
                for item in items:
                    # Draw bounding box
                    pts = np.array(item['bbox'], np.int32)
                    cv2.polylines(debug_img, [pts], True, color, 2)
                    # Draw text
                    cv2.putText(debug_img, f"{item['row_cluster']}-{item['text'][:10]}", 
                               (int(item['x_min']), int(item['y_min'])-5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Save debug visualization
            debug_path = os.path.join(self.debug_dir, f"page_{page_number}_table_{table_idx}_ocr_rows.jpg")
            cv2.imwrite(debug_path, debug_img)
            
            # Define columns based on horizontal positions
            # First detect column regions by analyzing the distribution of text items
            column_boundaries = self.detect_column_boundaries(clustered_items)
            self.logger.info(f"Detected {len(column_boundaries) - 1} potential columns")
            
            # Find header row (typically the first row)
            if rows_by_cluster:
                # Sort clusters by vertical position (top to bottom)
                sorted_clusters = sorted(rows_by_cluster.keys(), 
                                       key=lambda c: min(item['center_y'] for item in rows_by_cluster[c]))
                
                # Process all rows including header
                all_rows = []
                
                # Improved: Try to detect Dutch headers in the first row
                if sorted_clusters:
                    first_cluster = sorted_clusters[0]
                    header_items = rows_by_cluster[first_cluster]
                    
                    # Try to detect Dutch headers
                    header_data = {}
                    dutch_header_detected = False
                    
                    # Check for Dutch header patterns
                    for item in header_items:
                        text_lower = item['text'].lower()
                        
                        # Check for basic Dutch headers
                        if 'naam' in text_lower:
                            header_data['naam'] = item['text']
                            dutch_header_detected = True
                        elif 'dienst' in text_lower or 'nummer' in text_lower:
                            header_data['dienst-nummer'] = item['text']
                            dutch_header_detected = True
                        elif 'org' in text_lower or 'eenheid' in text_lower:
                            header_data['org.eenheid'] = item['text']
                            dutch_header_detected = True
                        elif any(day in text_lower for day in ['maandag', 'ma']):
                            header_data['maandag'] = item['text']
                            dutch_header_detected = True
                        elif any(day in text_lower for day in ['dinsdag', 'di']):
                            header_data['dinsdag'] = item['text']
                            dutch_header_detected = True
                        elif any(day in text_lower for day in ['woensdag', 'wo']):
                            header_data['woensdag'] = item['text']
                            dutch_header_detected = True
                        elif any(day in text_lower for day in ['donderdag', 'do']):
                            header_data['donderdag'] = item['text']
                            dutch_header_detected = True
                        elif any(day in text_lower for day in ['vrijdag', 'vr']):
                            header_data['vrijdag'] = item['text']
                            dutch_header_detected = True
                        elif any(day in text_lower for day in ['zaterdag', 'za']):
                            header_data['zaterdag'] = item['text']
                            dutch_header_detected = True
                        elif any(day in text_lower for day in ['zondag', 'zo']):
                            header_data['zondag'] = item['text']
                            dutch_header_detected = True
                    
                    # If no Dutch headers detected, use column index
                    if not dutch_header_detected:
                        for i, item in enumerate(header_items):
                            header_data[f"col_{i}"] = item['text']
                    
                    if header_data:
                        all_rows.append(header_data)
                
                # Process data rows
                for cluster in sorted_clusters[1:]:  # Skip header cluster
                    row_items = rows_by_cluster[cluster]
                    row_data = {}
                    
                    # Assign text to columns based on Dutch headers if detected
                    if dutch_header_detected:
                        # Create item positions for Dutch headers
                        header_positions = []
                        for key, text in header_data.items():
                            # Find position in header items
                            for item in header_items:
                                if item['text'] == text:
                                    header_positions.append((key, item['center_x']))
                                    break
                        
                        # Assign text based on closest header
                        for item in row_items:
                            if not header_positions:
                                break
                                
                            # Find closest header
                            closest_header = min(header_positions, key=lambda h: abs(h[1] - item['center_x']))
                            header_key = closest_header[0]
                            
                            # Add to row data
                            if header_key in row_data:
                                row_data[header_key] += " " + item['text']
                            else:
                                row_data[header_key] = item['text']
                    else:
                        # Assign based on column boundaries
                        for item in row_items:
                            col_idx = self.get_column_index(item['center_x'], column_boundaries)
                            col_key = f"col_{col_idx}"
                            
                            # Append value if column already has data (handles multi-part text in same column)
                            if col_key in row_data:
                                row_data[col_key] += " " + item['text']
                            else:
                                row_data[col_key] = item['text']
                    
                    if row_data:
                        all_rows.append(row_data)
                
                # Apply Dutch semantic recognition if headers were not detected
                if not dutch_header_detected and all_rows:
                    enhanced_rows = self.dutch_semantic_entity_recognition(all_rows)
                    if enhanced_rows and len(enhanced_rows) >= len(all_rows):
                        all_rows = enhanced_rows
                
                self.logger.info(f"OCR fallback extracted {len(all_rows)} rows")
                return all_rows
            else:
                self.logger.warning("No valid rows detected")
                return []
                
        except Exception as e:
            self.logger.error(f"Error extracting table with OCR fallback: {str(e)}")
            return []
    
    def simple_row_clustering(self, items, distance_threshold):
        """
        A simple row clustering algorithm based on y-coordinates.
        
        Args:
            items: List of text items with 'center_y' coordinates
            distance_threshold: Maximum distance for items to be considered in the same row
            
        Returns:
            The input items with 'row_cluster' field added
        """
        if not items:
            return items
        
        # Sort items by vertical position
        sorted_items = sorted(items, key=lambda item: item['center_y'])
        
        # Assign clusters
        current_cluster = 0
        sorted_items[0]['row_cluster'] = current_cluster
        
        # Process remaining items
        for i in range(1, len(sorted_items)):
            # Get vertical distance from previous item
            prev_y = sorted_items[i-1]['center_y']
            curr_y = sorted_items[i]['center_y']
            distance = abs(curr_y - prev_y)
            
            # If distance exceeds threshold, start a new cluster
            if distance > distance_threshold:
                current_cluster += 1
            
            # Assign cluster
            sorted_items[i]['row_cluster'] = current_cluster
        
        return sorted_items
    
    def detect_column_boundaries(self, text_items):
        """
        Detect column boundaries based on the distribution of text items.
        
        Args:
            text_items: List of text items with coordinates
            
        Returns:
            List of x-coordinates representing column boundaries
        """
        if not text_items:
            return [0, 1000]  # Default fallback
        
        # Extract x-coordinates
        x_centers = [item['center_x'] for item in text_items]
        x_lefts = [item['x_min'] for item in text_items]
        x_rights = [item['x_max'] for item in text_items]
        
        # Find min and max x-coordinates
        min_x = min(x_lefts)
        max_x = max(x_rights)
        
        # Simple approach: divide into 10 equal columns (can be improved)
        num_cols = 10
        boundaries = [min_x + i * (max_x - min_x) / num_cols for i in range(num_cols + 1)]
        
        return boundaries
    
    def get_column_index(self, x_coord, column_boundaries):
        """
        Get column index for a given x-coordinate.
        
        Args:
            x_coord: X-coordinate
            column_boundaries: List of x-coordinates representing column boundaries
            
        Returns:
            Column index
        """
        for i in range(len(column_boundaries) - 1):
            if column_boundaries[i] <= x_coord < column_boundaries[i + 1]:
                return i
        return len(column_boundaries) - 2  # Last column
    
    def calculate_horizontal_overlap(self, item1: Dict, item2: Dict) -> float:
        """
        Calculate horizontal overlap between two text items.
        
        Args:
            item1: First text item
            item2: Second text item
            
        Returns:
            Overlap score (higher is better)
        """
        # If items are perfectly aligned
        if item1['center_x'] >= item2['x_min'] and item1['center_x'] <= item2['x_max']:
            return 1.0
            
        # Calculate distance between center of item1 and center of item2
        dist = abs(item1['center_x'] - item2['center_x'])
        
        # Normalize by the width of the widest item
        max_width = max(item1['width'], item2['width'])
        if max_width == 0:
            return 0
            
        normalized_dist = dist / max_width
        
        # Convert to overlap score (1 means perfect alignment, 0 means no overlap)
        return max(0, 1 - normalized_dist)
    
    def smart_post_processing(self, rows_data: List[Dict], page_number: int) -> List[Dict]:
        """
        Apply smart post-processing to fix common issues and format data correctly.
        
        Args:
            rows_data: List of extracted rows
            page_number: Page number for debugging
            
        Returns:
            List of employee records with corrected data
        """
        try:
            start_time = time.time()
            self.logger.info(f"[5/6] Page {page_number}: Smart post-processing of {len(rows_data)} rows")
            
            # If no data, return empty list
            if not rows_data:
                return []
            
            # Step 1: Infer column types from headers
            column_types = self.infer_column_types(rows_data[0] if rows_data else {})
            self.logger.info(f"Inferred column types: {column_types}")
            
            # Step 2: Process rows to employee records
            employee_records = []
            for i, row in enumerate(rows_data):
                # Skip the header row
                if i == 0:
                    continue
                    
                # Extract employee information
                employee = self.extract_employee_from_row(row, column_types)
                
                if employee:
                    employee_records.append(employee)
            
            # Step 3: Fix fragmented names and misclassified data
            corrected_records = []
            
            # First pass: flag potential fragments
            for i, record in enumerate(employee_records):
                # Check if name ends with comma (potential fragment)
                if record['naam'] and record['naam'].strip().endswith(','):
                    record['is_name_fragment'] = True
                else:
                    record['is_name_fragment'] = False
                
                # Flag potential number fragments (names that are just numbers)
                if record['naam'] and re.match(r'^\d+$', record['naam'].strip()):
                    record['is_nummer_fragment'] = True
                else:
                    record['is_nummer_fragment'] = False
                
                # Flag potential schedule fragments (names that look like times)
                if record['naam'] and re.search(r'\d{1,2}:\d{2}', record['naam']):
                    record['is_schedule_fragment'] = True
                else:
                    record['is_schedule_fragment'] = False
            
            # Second pass: merge fragments and fix misclassifications
            skip_indices = set()
            for i, record in enumerate(employee_records):
                if i in skip_indices:
                    continue
                
                corrected_record = record.copy()
                
                # Fix name fragments
                if record.get('is_name_fragment') and i + 1 < len(employee_records):
                    next_record = employee_records[i + 1]
                    
                    # Check if the next record looks like initials or surname (2-4 uppercase letters)
                    if re.match(r'^[A-Z]{2,4}', next_record.get('naam', '').strip()):
                        corrected_record['naam'] = f"{record['naam']} {next_record['naam']}"
                        skip_indices.add(i + 1)
                
                # Fix number fragments (names that are just numbers)
                if record.get('is_nummer_fragment'):
                    corrected_record['dienst-nummer'] = record['naam']
                    
                    # If there's a previous record with name and no number, assign this number to it
                    if i > 0 and not employee_records[i-1]['dienst-nummer'] and employee_records[i-1]['naam']:
                        corrected_records[-1]['dienst-nummer'] = record['naam']
                        continue  # Skip adding this as a separate record
                
                # Fix schedule fragments (names that look like times)
                if record.get('is_schedule_fragment'):
                    # Don't add this as a separate employee record
                    if i > 0 and employee_records[i-1]['naam']:
                        # If the previous record already has a schedule, try to add to it
                        if corrected_records and corrected_records[-1]['schedule']:
                            # Add to the last day's schedule
                            if corrected_records[-1]['schedule']:
                                last_day = corrected_records[-1]['schedule'][-1]
                                parsed_entries = self.parse_schedule_entry(record['naam'])
                                if parsed_entries:
                                    last_day['schedule'].extend(parsed_entries)
                        continue
                
                # Remove processing flags
                for flag in ['is_name_fragment', 'is_nummer_fragment', 'is_schedule_fragment']:
                    if flag in corrected_record:
                        del corrected_record[flag]
                
                # Handle mixed number and name cases (extract number from name if necessary)
                if ' ' in corrected_record['naam'] and any(part.isdigit() and len(part) > 5 for part in corrected_record['naam'].split()):
                    parts = corrected_record['naam'].split()
                    for part in parts:
                        if part.isdigit() and len(part) > 5:
                            # This is likely a dienst-nummer
                            corrected_record['dienst-nummer'] = part
                            # Remove ID from name
                            corrected_record['naam'] = ' '.join([p for p in parts if p != part])
                
                # Clean up the record data
                corrected_record['naam'] = corrected_record['naam'].strip()
                corrected_record['dienst-nummer'] = corrected_record['dienst-nummer'].strip()
                corrected_record['org.eenheid'] = corrected_record['org.eenheid'].strip()
                
                # Validate and add to final records
                if self.is_valid_employee_record(corrected_record):
                    corrected_records.append(corrected_record)
            
            elapsed = time.time() - start_time
            self.logger.info(f"[5/6] Post-processing complete in {elapsed:.2f}s. Found {len(corrected_records)} valid records")
            return corrected_records
            
        except Exception as e:
            self.logger.error(f"[FAILED] [5/6] Error in smart post-processing: {str(e)}")
            return rows_data
    
    def validate_and_deduplicate(self, employee_records: List[Dict]) -> List[Dict]:
        """
        Validate and deduplicate employee records across all pages.
        Ensures every employee has entries for all 7 days of the week.
        
        Args:
            employee_records: List of employee records from all pages
            
        Returns:
            List of validated and deduplicated employee records with complete schedules
        """
        try:
            start_time = time.time()
            self.logger.info(f"[6/6] Validating and deduplicating {len(employee_records)} employee records")
            
            # Group employees by name for deduplication
            employees_by_name = defaultdict(list)
            for record in employee_records:
                # Use lowercase name as key for case-insensitive matching
                name_key = record['naam'].lower()
                employees_by_name[name_key].append(record)
            
            # Merge and deduplicate records
            unique_employees = []
            
            # Define day order and Dutch names for all 7 days
            day_order = {
                'maandag': 1, 'dinsdag': 2, 'woensdag': 3, 'donderdag': 4, 
                'vrijdag': 5, 'zaterdag': 6, 'zondag': 7
            }
            
            for name_key, records in employees_by_name.items():
                if not records:
                    continue
                
                # Start with the first record as base
                merged_record = records[0].copy()
                
                # Merge additional data from other records with the same name
                for record in records[1:]:
                    # Take non-empty dienst-nummer if current is empty
                    if not merged_record['dienst-nummer'] and record['dienst-nummer']:
                        merged_record['dienst-nummer'] = record['dienst-nummer']
                    
                    # Take non-empty org.eenheid if current is empty
                    if not merged_record['org.eenheid'] and record['org.eenheid']:
                        merged_record['org.eenheid'] = record['org.eenheid']
                    
                    # Merge schedule data
                    for day_entry in record['schedule']:
                        # Check if this day is already in the merged record
                        day_exists = False
                        for existing_day in merged_record['schedule']:
                            if existing_day['day'] == day_entry['day'] and existing_day['date'] == day_entry['date']:
                                # Merge the schedules for this day
                                existing_day['schedule'].extend(day_entry['schedule'])
                                day_exists = True
                                break
                        
                        # If this day isn't already in the merged record, add it
                        if not day_exists:
                            merged_record['schedule'].append(day_entry)
                
                # Ensure each employee has a dienst-nummer (generate one if missing)
                if not merged_record['dienst-nummer']:
                    merged_record['dienst-nummer'] = f"EMP{len(unique_employees):03d}"
                
                # Create a dictionary to track which days are present
                days_present = {day: False for day in day_order.keys()}
                dates_by_day = {}
                
                # Mark which days are present in the schedule
                for entry in merged_record['schedule']:
                    day = entry['day']
                    days_present[day] = True
                    dates_by_day[day] = entry['date']
                
                # Add missing days with "Off" schedule
                for day, present in days_present.items():
                    if not present:
                        # Try to determine a reasonable date for the missing day
                        # If we have dates for other days, try to infer this one
                        date = ""
                        if dates_by_day:
                            # Use the date format from another day if available
                            sample_date = next(iter(dates_by_day.values()))
                            if sample_date:
                                # Just use the sample date as a placeholder
                                # In a production environment, we might calculate the actual date
                                date = sample_date
                        
                        # Add the missing day with "Off" schedule
                        merged_record['schedule'].append({
                            'day': day,
                            'date': date,
                            'schedule': ["Off"]
                        })
                
                # Sort schedule by day order
                merged_record['schedule'].sort(key=lambda x: day_order.get(x['day'], 99))
                
                unique_employees.append(merged_record)
            
            # Final validation of all records
            final_employees = [emp for emp in unique_employees if self.is_valid_employee_record(emp)]
            
            # Log the final structure to verify
            if final_employees:
                self.logger.info(f"Sample employee record structure: {json.dumps(final_employees[0], indent=2)}")
            
            elapsed = time.time() - start_time
            self.logger.info(f"[6/6] Validation complete in {elapsed:.2f}s. Final count: {len(final_employees)} employees")
            return final_employees
            
        except Exception as e:
            self.logger.error(f"[FAILED] [6/6] Error in validation and deduplication: {str(e)}")
            return employee_records
    
    def infer_column_types(self, headers: Dict) -> Dict:
        """
        Infer column types based on Dutch header text.
        
        Args:
            headers: Row data with headers
            
        Returns:
            Dictionary mapping column names to types with additional date info
        """
        column_types = {}
        
        # Exact Dutch header keywords for matching
        naam_keywords = ['naam']
        dienst_nummer_keywords = ['dienst nummer', 'dienstnummer', 'dienst nr', 'nr', 'nummer']
        org_eenheid_keywords = ['org.eenheid', 'org eenheid', 'orgeenheid', 'org', 'eenheid']
        
        # Day keywords with date extraction regex pattern
        day_keywords = {
            'maandag': ['maandag'],
            'dinsdag': ['dinsdag'],
            'woensdag': ['woensdag'],
            'donderdag': ['donderdag'],
            'vrijdag': ['vrijdag'],
            'zaterdag': ['zaterdag'],
            'zondag': ['zondag']
        }
        
        # Date extraction pattern: looks for date formats like 07/04/2024 or similar
        date_pattern = re.compile(r'(\d{1,2}[/\.-]\d{1,2}[/\.-]\d{2,4})')
        
        # Process each header
        for header, value in headers.items():
            header_text = str(value).lower().strip()
            
            # Check for "Naam" column
            if any(keyword in header_text for keyword in naam_keywords):
                column_types[header] = {'type': 'naam'}
                continue
            
            # Check for "Dienst Nummer" column
            if any(keyword in header_text for keyword in dienst_nummer_keywords):
                column_types[header] = {'type': 'dienst-nummer'}
                continue
            
            # Check for "Org.Eenheid" column
            if any(keyword in header_text for keyword in org_eenheid_keywords):
                column_types[header] = {'type': 'org.eenheid'}
                continue
            
            # Check for day columns and extract dates
            for day, keywords in day_keywords.items():
                if any(keyword in header_text for keyword in keywords):
                    # Try to extract date from header
                    date_match = date_pattern.search(header_text)
                    date_value = date_match.group(1) if date_match else ""
                    
                    column_types[header] = {
                        'type': 'day',
                        'day': day,
                        'date': date_value
                    }
                    break
        
        # If no columns were identified, log a warning
        if not column_types:
            self.logger.warning("Could not identify Dutch column types from headers, using positional inference")
            
            # Simple positional inference as fallback
            cols = list(headers.keys())
            if len(cols) >= 1:
                column_types[cols[0]] = {'type': 'naam'}
            if len(cols) >= 2:
                column_types[cols[1]] = {'type': 'dienst-nummer'}
            if len(cols) >= 3:
                column_types[cols[2]] = {'type': 'org.eenheid'}
            
            # Remaining columns are likely days of the week
            days = ['maandag', 'dinsdag', 'woensdag', 'donderdag', 'vrijdag', 'zaterdag', 'zondag']
            for i, col in enumerate(cols[3:10]):  # Limit to 7 days
                if i < len(days):
                    column_types[col] = {'type': 'day', 'day': days[i], 'date': ''}
        
        return column_types
    
    def extract_employee_from_row(self, row: Dict, column_types: Dict) -> Dict:
        """
        Extract employee information from a row based on Dutch column types.
        
        Args:
            row: Row data
            column_types: Dictionary mapping column names to types
            
        Returns:
            Employee dictionary in the requested format
        """
        employee = {
            'naam': '',
            'dienst-nummer': '',
            'org.eenheid': '',
            'schedule': []
        }
        
        # Extract data from basic employee info columns
        for column, value in row.items():
            if not value or not column in column_types:
                continue
            
            column_info = column_types[column]
            column_type = column_info.get('type', '')
            
            # Basic fields
            if column_type == 'naam':
                employee['naam'] = str(value).strip()
            elif column_type == 'dienst-nummer':
                employee['dienst-nummer'] = str(value).strip()
            elif column_type == 'org.eenheid':
                employee['org.eenheid'] = str(value).strip()
        
        # Extract schedule data for each day
        for column, value in row.items():
            if not value or not column in column_types:
                continue
            
            column_info = column_types[column]
            if column_info.get('type') == 'day':
                day = column_info.get('day', '')
                date = column_info.get('date', '')
                schedule_value = str(value).strip()
                
                if schedule_value and schedule_value.lower() != 'off':
                    # Parse the schedule text to extract shift information
                    schedule_entries = self.parse_schedule_entry(schedule_value)
                    
                    # Add to employee schedule
                    if schedule_entries:
                        employee['schedule'].append({
                            'day': day,
                            'date': date,
                            'schedule': schedule_entries
                        })
        
        return employee
    
    def parse_schedule_entry(self, schedule_text: str) -> List[str]:
        """
        Parse a schedule entry text to extract shift details.
        
        Args:
            schedule_text: Text containing schedule information
            
        Returns:
            List of schedule entries with service type and times
        """
        # Clean the input text
        text = schedule_text.strip()
        if not text or text.lower() == 'off':
            return []
        
        # Split entries if multiple are present (e.g., by newlines or semicolons)
        entries = []
        
        # Check for common separators
        if '\n' in text:
            parts = text.split('\n')
        elif ';' in text:
            parts = text.split(';')
        elif '/' in text:
            parts = text.split('/')
        else:
            parts = [text]  # Single entry
        
        # Process each part
        for part in parts:
            part = part.strip()
            if not part:
                continue
            
            # Try to identify common patterns:
            # 1. DIENST 07:00 16:30
            # 2. [Rust] 13:00 22:30
            # 3. [Vr.zondag] 00:00 24:00
            
            # Check for bracketed service type
            if '[' in part and ']' in part:
                service_match = re.search(r'\[(.*?)\]', part)
                service_type = service_match.group(1) if service_match else ""
                
                # Extract times
                times_match = re.search(r'(\d{1,2}:\d{2})\s+(\d{1,2}:\d{2})', part)
                if times_match:
                    start_time = times_match.group(1)
                    end_time = times_match.group(2)
                    entries.append(f"[{service_type}] {start_time} {end_time}")
                else:
                    entries.append(part)  # Keep as is if no time pattern found
            
            # Check for service followed by times
            elif re.search(r'(DIENST|RUST)\s+\d{1,2}:\d{2}\s+\d{1,2}:\d{2}', part, re.IGNORECASE):
                entries.append(part)
            
            # Just times without service type
            elif re.search(r'(\d{1,2}:\d{2})\s+(\d{1,2}:\d{2})', part):
                times_match = re.search(r'(\d{1,2}:\d{2})\s+(\d{1,2}:\d{2})', part)
                start_time = times_match.group(1)
                end_time = times_match.group(2)
                entries.append(f"DIENST {start_time} {end_time}")  # Default to DIENST
            
            # Anything else
            else:
                entries.append(part)
        
        return entries
    
    def is_valid_name(self, name: str) -> bool:
        """
        Check if a name is valid.
        
        Args:
            name: Name to check
            
        Returns:
            True if valid, False otherwise
        """
        if not name or len(name.strip()) < 2:
            return False
            
        # Must contain at least one letter
        if not any(c.isalpha() for c in name):
            return False
            
        # Cannot be just a time pattern
        if re.match(r'^\d{1,2}:\d{2}(\s+\d{1,2}:\d{2})?', name.strip()):
            return False
            
        # Cannot be just a day or status keyword
        day_keywords = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 
                       'saturday', 'sunday', 'maandag', 'dinsdag', 'woensdag',
                       'donderdag', 'vrijdag', 'zaterdag', 'zondag']
        status_keywords = ['dienst', 'service', 'off', 'vrij', 'free', 'rust', 'rest']
        
        name_lower = name.lower().strip()
        if any(keyword == name_lower for keyword in day_keywords + status_keywords):
            return False
            
        return True
    
    def is_valid_employee_record(self, record: Dict) -> bool:
        """
        Check if an employee record is valid in the new format.
        
        Args:
            record: Employee record to check
            
        Returns:
            True if valid, False otherwise
        """
        # Must have a valid name
        if not self.is_valid_name(record['naam']):
            return False
        
        # Valid if has name and at least one other valid field
        has_dienst_nummer = bool(record['dienst-nummer'])
        has_org_eenheid = bool(record['org.eenheid'])
        has_schedule = bool(record['schedule'])
        
        return bool(record['naam'] and (has_dienst_nummer or has_org_eenheid or has_schedule))