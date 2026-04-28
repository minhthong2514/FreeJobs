import os
import re

# Đường dẫn của bạn
folder_path = "/home/minhthong/Desktop/code/farmbot/models/datasets/potatoes"

def flexible_rename_tool(directory=folder_path, prefix="potato", offset=100, start_at=251):
    # Các định dạng ảnh hỗ trợ
    valid_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')
    
    # Regex để nhận diện đúng định dạng: prefix-số.ext
    pattern = re.compile(rf'^{re.escape(prefix)}-(\d+)(\.(?:jpg|jpeg|png|gif|bmp|webp))$')

    files = os.listdir(directory)
    files_to_shift = []
    other_files = []

    # 1. Phân loại file: Chỉ chọn file có index >= start_at
    for filename in files:
        if not filename.lower().endswith(valid_extensions):
            continue
            
        match = pattern.match(filename.lower())
        if match:
            idx = int(match.group(1))
            if idx >= start_at:
                files_to_shift.append({
                    'old_name': filename,
                    'index': idx,
                    'ext': match.group(2).lower()
                })
            else:
                # Những file nhỏ hơn 100 giữ nguyên
                continue 
        else:
            # File ảnh lạ (không phải prefix-X)
            other_files.append(filename)

    # 2. Xử lý tịnh tiến (Offset) - Sắp xếp giảm dần để tránh đè tên
    files_to_shift.sort(key=lambda x: x['index'], reverse=True)
    
    print(f"--- Đang tịnh tiến index cho '{prefix}' từ {start_at} trở đi (Cộng thêm {offset}) ---")
    
    max_index_found = 0
    for file_info in files_to_shift:
        new_index = file_info['index'] + offset
        new_name = f"{prefix}-{new_index}{file_info['ext']}"
        
        if new_index > max_index_found:
            max_index_found = new_index
            
        old_path = os.path.join(directory, file_info['old_name'])
        new_path = os.path.join(directory, new_name)
        
        os.rename(old_path, new_path)
        print(f"Đã dịch chuyển: {file_info['old_name']} -> {new_name}")

    # 3. Đổi tên các file mới (không theo định dạng prefix-X)
    # Bắt đầu đánh số từ max_index sau khi đã offset
    if other_files:
        print(f"\n--- Đang xử lý {len(other_files)} file ảnh mới ---")
        other_files.sort()
        for filename in other_files:
            max_index_found += 1
            ext = os.path.splitext(filename)[1].lower()
            new_name = f"{prefix}-{max_index_found}{ext}"
            
            os.rename(os.path.join(directory, filename), os.path.join(directory, new_name))
            print(f"Đã thêm mới: {filename} -> {new_name}")

    print(f"\n--- HOÀN TẤT ---")

if __name__ == "__main__":
    # offset=50: 100->150, 101->151...
    # start_at=100: Chỉ những file từ background-100 trở đi mới bị đổi
    flexible_rename_tool()