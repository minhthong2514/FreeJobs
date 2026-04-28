import os
import re

folder_path = "/home/minhthong/Desktop/code/farmbot/models/datasets/background"
def rename_background_images(directory=folder_path):
    # Các định dạng ảnh hỗ trợ
    valid_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')
    
    # 1. Lấy danh sách tất cả các file trong thư mục
    files = os.listdir(directory)
    
    # 2. Tìm số thứ tự lớn nhất hiện có của các file "background-X"
    current_max_index = 0
    bg_pattern = re.compile(r'^background-(\d+)\.(?:jpg|jpeg|png|gif|bmp|webp)$')
    
    for filename in files:
        match = bg_pattern.match(filename.lower())
        if match:
            index = int(match.group(1))
            if index > current_max_index:
                current_max_index = index
                
    print(f"--- Số thứ tự lớn nhất hiện tại là: {current_max_index} ---")

    # 3. Rename các file chưa đúng định dạng
    count = 0
    for filename in files:
        # Kiểm tra nếu là file ảnh và KHÔNG bắt đầu bằng 'background-'
        if filename.lower().endswith(valid_extensions) and not filename.lower().startswith("background-"):
            current_max_index += 1
            extension = os.path.splitext(filename)[1]
            new_name = f"background-{current_max_index}{extension}"
            
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_name)
            
            try:
                os.rename(old_path, new_path)
                print(f"Đã đổi: {filename} -> {new_name}")
                count += 1
            except Exception as e:
                print(f"Lỗi khi đổi tên {filename}: {e}")

    if count == 0:
        print("Không có file nào cần đổi tên.")
    else:
        print(f"--- Hoàn tất! Đã đổi tên {count} file. ---")

if __name__ == "__main__":
    # Bạn có thể thay "." bằng đường dẫn thư mục cụ thể, ví dụ: "C:/Users/Images"
    rename_background_images()