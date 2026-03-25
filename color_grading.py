from PIL import Image
import os

def convert_palette_to_rgb(image_folder_path, output_folder_path):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    # 获取文件夹中的所有文件
    files = os.listdir(image_folder_path)
    converted_count = 0

    # 遍历文件夹中的每个文件
    for file in files:
        file_path = os.path.join(image_folder_path, file)
        if os.path.isfile(file_path):
            try:
                # 打开图像
                with Image.open(file_path) as img:
                    # 检查图像模式
                    if img.mode == 'P':
                        print(f"正在转换调色板模式图像: {file}")
                        # 转换为 RGB 模式
                        rgb_image = img.convert('RGB')
                        # 保存转换后的图像
                        output_path = os.path.join(output_folder_path, file)
                        rgb_image.save(output_path)
                        converted_count += 1
            except IOError:
                # 如果文件不是图像文件，跳过
                continue

    return converted_count

# 指定图片文件夹路径和输出文件夹路径
image_folder_path = r'./gossipcop/gossip_test'  # 替换为你的图片文件夹路径
output_folder_path = r'./gossipcop/gossip_test' # 替换为你的输出文件夹路径

# 调用函数并输出结果
converted_count = convert_palette_to_rgb(image_folder_path, output_folder_path)
print(f"已转换的调色板模式图像数量: {converted_count}")