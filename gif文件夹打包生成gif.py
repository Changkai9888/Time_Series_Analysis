import imageio
import os
def save_gif(num):
    # 指定 PNG 图像的文件夹路径和文件名模板
    image_folder = './gif/'
    image_name_template = '{}.png'

    # 生成 PNG 文件名列表，并按照数字顺序排序
    image_files = [os.path.join(image_folder, image_name_template.format('{:03d}'.format(i))) for i in range(1,num)]
    image_files.sort()

    '''# 使用 imageio 打开 PNG 图像，将它们添加到 GIF 动画中
    with imageio.get_writer('animation.gif', mode='I') as writer:
        for filename in image_files:
            image = imageio.imread(filename)
            writer.append_data(image)'''
    # 设置 GIF 动画的帧率为每秒 5 帧
    fps = 4

    # 使用 imageio 打开 PNG 图像，将它们添加到 GIF 动画中
    images = [imageio.imread(filename) for filename in image_files]
    imageio.mimsave('.\\gif\\000000.gif', images, fps=fps)
