import os
from PIL import Image

image_path = "/Users/zfwang/project/machinelearning/deeplearning/deeplearning/src/pillow_src/images"
image_name = "lena"

# try:
#     with Image.open(os.path.join(image_path, image_name + ".png")) as im:
#         # 图像格式
#         print(im.format)

#         # 图像尺寸
#         print(im.size)

#         # 图像模式
#         print(im.mode)

#         # 图像打印
#         im.show()
# except OSError:
#     print(f"cannot open {image_name}")


# try:
#     with Image.open(os.path.join(image_path, image_name + ".png")) as im:
#         im.save(os.path.join(image_path, image_name + ".jpg"))
# except OSError as e:
#     print(f"Can't open {image_name}")


# try:
#     with Image.open(os.path.join(image_path, image_name + ".jpg")) as im:
#         # 图像格式
#         print(im.format)

#         # 图像尺寸
#         print(im.size)

#         # 图像模式
#         print(im.mode)

#         # 图像打印
#         im.show()
# except OSError:
#     print(f"cannot open {image_name}")



# try:
#     size = (128, 128)
#     with Image.open(os.path.join(image_path, image_name + ".png")) as im:
#         im.thumbnail(size)
#         im.save(os.path.join(image_path, image_name + ".JPEG"))
# except IOError as e:
#     print(f"Can't open {image_name}")


# try:
#     with Image.open(os.path.join(image_path, image_name + ".png")) as im:
#         print(image_name, im.format, f"{im.size}x{im.mode}")
# except IOError as e:
#     print(f"Can't open {image_name}")


# try:
#     box = (100, 100, 200, 200)
#     with Image.open(os.path.join(image_path, image_name + ".png")) as im:
#         region = im.crop(box)
#         region.show()
#         region.save(os.path.join(image_path, image_name + "_region" + ".png"))
# except IOError as e:
#     print(f"Can't open {image_name}")


# try:
#     box = (100, 100, 200, 200)
#     with Image.open(os.path.join(image_path, image_name + ".png")) as im:
#         region = im.crop(box)
#         region = region.transpose(Image.ROTATE_180)
#         im.paste(region, box)
#         im.save(os.path.join(image_path, image_name + "_region_paste" + ".png"))
# except IOError as e:
#     print(f"Can't open {image_name}")





# def roll(image, delta):
#     """Roll an image sideways"""
#     xsize, ysize = image.size
#     delta = delta % xsize
#     print(xsize)
#     print(delta)
#     if delta == 0:
#         return image
#     part1 = image.crop((0, 0, delta, ysize))
#     part2 = image.crop((delta, 0, xsize, ysize))
#     image.paste(part1, (xsize - delta, 0, xsize, ysize))
#     image.paste(part2, (0, 0, xsize - delta, ysize))

#     return image

# try:
#     with Image.open(os.path.join(image_path, image_name + ".png")) as im:
#         im = roll(im, 100000)
#         # 图像打印
#         im.save(os.path.join(image_path, image_name + "_roll" + ".png"))
# except OSError:
#     print(f"cannot open {image_name}")


# try:
#     with Image.open(os.path.join(image_path, image_name + ".png")) as im:
#         r, g, b = im.split()
#         im = Image.merge("RGB", (b, r, g))
#         im.save(os.path.join(image_path, image_name + "_merge_brg" + ".png"))
# except OSError:
#     print(f"cannot open {image_name}")


# try:
#     with Image.open(os.path.join(image_path, image_name + ".png")) as im:
#         out = im.resize((1000, 1000))
#         out.save(os.path.join(image_path, image_name + "_resize" + ".png"))
# except OSError:
#     print(f"cannot open {image_name}")

# try:
#     with Image.open(os.path.join(image_path, image_name + ".png")) as im:
#         out = im.rotate(45)
#         out.save(os.path.join(image_path, image_name + "_rotate" + ".png"))
# except OSError:
#     print(f"cannot open {image_name}")


try:
    with Image.open(os.path.join(image_path, image_name + ".png")) as im:
        out1 = im.transpose(Image.FLIP_LEFT_RIGHT)
        out2 = im.transpose(Image.FLIP_TOP_BOTTOM)
        out3 = im.transpose(Image.ROTATE_90)
        out4 = im.transpose(Image.ROTATE_180)
        out5 = im.transpose(Image.ROTATE_270)
        out1.save(os.path.join(image_path, image_name + "_rotate_1" + ".png"))
        out2.save(os.path.join(image_path, image_name + "_rotate_2" + ".png"))
        out3.save(os.path.join(image_path, image_name + "_rotate_3" + ".png"))
        out4.save(os.path.join(image_path, image_name + "_rotate_4" + ".png"))
        out5.save(os.path.join(image_path, image_name + "_rotate_5" + ".png"))
except OSError:
    print(f"cannot open {image_name}")