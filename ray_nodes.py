# Merged ComfyUI Custom Nodes
# Auto-generated file

import torch
import numpy as np
from typing import Tuple  # 添加这行
from PIL import Image
import cv2
import os
import re
import time
import comfy.utils


# From bracketed-tag-merger.py
class BracketedTagIndexMerger:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tag_string": ("STRING", {"forceInput": True}),
                "index_string": ("STRING", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "merge_tags_and_indices"
    CATEGORY = "image"

    def merge_tags_and_indices(self, tag_string: str, index_string: str) -> tuple[str]:
        # 确保输入是字符串
        if isinstance(tag_string, list):
            tag_string = tag_string[0] if tag_string else ""
        if isinstance(index_string, list):
            index_string = index_string[0] if index_string else ""
            
        # 清理输入字符串
        tag_string = tag_string.strip()
        
        # 如果输入是字符串形式的列表，将其转换为实际的列表
        import ast
        try:
            if tag_string.startswith('[') and tag_string.endswith(']'):
                tag_list = ast.literal_eval(tag_string)
            else:
                tag_list = [tag_string]
        except:
            # 如果解析失败，按原样处理
            tag_list = [tag_string]
        
        # 处理索引
        indices = [idx.strip() for idx in index_string.split(',') if idx.strip()]
        
        # 确保索引和tag组数量匹配
        if len(indices) != len(tag_list):
            print(f"Warning: Number of indices ({len(indices)}) does not match number of tag groups ({len(tag_list)})")
            print(f"Indices: {indices}")
            print(f"Tags: {tag_list}")
        
        # 合并索引和tag组
        result = []
        for idx, tags in zip(indices, tag_list):
            # 保持tags的原始格式，不进行额外的分割
            merged = f'"{idx}":"{tags}"'
            result.append(merged)
        
        final_string = ','.join(result)
        return (final_string,)

NODE_CLASS_MAPPINGS = {
    "BracketedTagIndexMerger": BracketedTagIndexMerger
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BracketedTagIndexMerger": "🏷️ Bracketed Tag-Index Merger"
}


# From florence2-tag-processor.py
class Florence2TagProcessor:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tag_input": ("STRING", {}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "process_tags"
    CATEGORY = "Tag Processing"

    def process_tags(self, tag_input: str) -> tuple[str]:
    # 如果 tag_input 是一个列表，将其转换为换行分隔的字符串
        if isinstance(tag_input, list):
            converted_text = "\n".join(tag_input)  # 直接用换行符拼接列表元素
        else:
            # 如果是字符串，按需要清理
            cleaned_text = tag_input.strip('[]"')
            converted_text = cleaned_text.replace('", "', '\n')
        
        return (converted_text,)

NODE_CLASS_MAPPINGS.update({
    "Florence2TagProcessor": Florence2TagProcessor
})

NODE_DISPLAY_NAME_MAPPINGS.update({
    "Florence2TagProcessor": "🏷️ Florence2 Tag Processor"
})


# From Image List Converter.py
class ImageListConverter:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image_list",)
    FUNCTION = "convert_to_image_list"
    CATEGORY = "Image Processing"

    def convert_to_image_list(self, image1=None, image2=None, image3=None, image4=None):
        image_list = []

        for image in [image1, image2, image3, image4]:
            if image is not None:
                if isinstance(image, torch.Tensor):
                    if image.dim() == 3:  # Single image
                        image_list.append(image.unsqueeze(0))
                    elif image.dim() == 4:  # Batch of images
                        image_list.extend([img.unsqueeze(0) for img in image])
                elif isinstance(image, list):  # List of images
                    for img in image:
                        if isinstance(img, torch.Tensor) and img.dim() == 3:
                            image_list.append(img.unsqueeze(0))

        if not image_list:
            # Return an empty tensor if no valid images were found
            return (torch.empty((0, 3, 1, 1)),)

        # Stack all images into a single tensor
        return (torch.cat(image_list, dim=0),)

NODE_CLASS_MAPPINGS.update({
    "ImageListConverter": ImageListConverter,
})

NODE_DISPLAY_NAME_MAPPINGS.update({
    "ImageListConverter": "🖼️Image List Converter"
})


# From image-selector-node.py
class ImageSelector:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "interval": ("INT", {
                    "default": 1, 
                    "min": 1,
                    "max": 100,
                    "step": 1
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("selected_images", "selected_indices")
    FUNCTION = "select_images"
    CATEGORY = "Image Processing"

    def select_images(self, images: torch.Tensor, interval: int) -> Tuple[torch.Tensor, str]:
        num_images = images.shape[0]
        selected_indices = list(range(0, num_images, interval))
        
        selected_images = images[selected_indices]
        
        indices_str = ",".join(map(str, selected_indices))
        
        return (selected_images, indices_str)

NODE_CLASS_MAPPINGS.update({
    "ImageSelector": ImageSelector
})

NODE_DISPLAY_NAME_MAPPINGS.update({
    "ImageSelector": "🖼️ Image Selector"
})


# From Mask Blackener.py
def pil2tensor(image: Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def tensor2pil(t_image: torch.Tensor) -> Image:
    return Image.fromarray(np.clip(255.0 * t_image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def mask_to_pil(mask) -> Image:
    if isinstance(mask, torch.Tensor):
        mask_np = mask.squeeze().cpu().numpy()
    elif isinstance(mask, np.ndarray):
        mask_np = mask
    else:
        raise TypeError("Unsupported mask type")
    mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8))
    return mask_pil

class MaskBlackener:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("blackened_image",)
    FUNCTION = 'apply_black_mask'
    CATEGORY = 'Image Processing'

    def apply_black_mask(self, image, mask):
        # Convert image tensor to PIL
        image_pil = tensor2pil(image.squeeze(0)).convert('RGB')
        
        # Convert mask to PIL
        mask_pil = mask_to_pil(mask).convert('L')
        
        # Create a black image of the same size
        black_image = Image.new('RGB', image_pil.size, (0, 0, 0))
        
        # Apply the mask: use the original image where mask is black, and black image where mask is white
        blackened_image = Image.composite(black_image, image_pil, mask_pil)
        
        # Convert the result back to tensor
        blackened_image_tensor = pil2tensor(blackened_image)
        
        return (blackened_image_tensor,)

NODE_CLASS_MAPPINGS.update({
    "MaskBlackener": MaskBlackener,
})

NODE_DISPLAY_NAME_MAPPINGS.update({
    "MaskBlackener": "🖤Mask Blackener"
})


# From mask-applier-fixed-v2.py
def pil2tensor(image: Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def tensor2pil(t_image: torch.Tensor) -> Image:
    return Image.fromarray(np.clip(255.0 * t_image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def mask_to_pil(mask) -> Image:
    if isinstance(mask, torch.Tensor):
        mask_np = mask.squeeze().cpu().numpy()
    elif isinstance(mask, np.ndarray):
        mask_np = mask
    else:
        raise TypeError("Unsupported mask type")
    mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8))
    return mask_pil

def apply_feather(mask_np: np.ndarray, feather_amount: int) -> np.ndarray:
    """Apply feathering effect to a mask using Gaussian blur."""
    if feather_amount <= 0:
        return mask_np

    # 确保mask是灰度图
    if len(mask_np.shape) > 2:
        mask_np = cv2.cvtColor(mask_np, cv2.COLOR_RGB2GRAY)

    # 转换为float32类型以保持精度
    mask_np = mask_np.astype(np.float32) / 255.0

    # 创建高斯核进行模糊
    # kernel_size必须是正奇数，且与feather_amount成正比
    kernel_size = max(3, feather_amount * 2 + 1)
    sigma = max(0.1, feather_amount / 3.0)  # sigma与feather_amount相关

    # 应用高斯模糊实现羽化效果
    feathered = cv2.GaussianBlur(mask_np, (kernel_size, kernel_size), sigma)

    # 重新缩放到0-255范围
    result = np.clip(feathered * 255, 0, 255)

    return result.astype(np.uint8)

class MaskApplierAndCombiner:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "masks": ("MASK",),
                "feather_amount": ("INT", {
                    "default": 0, 
                    "min": 0, 
                    "max": 100,
                    "step": 1
                }),
            },
            "optional": {}
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("combined_image",)
    FUNCTION = 'apply_masks_and_combine'
    CATEGORY = 'Image Processing'

    def apply_masks_and_combine(self, images, masks, feather_amount):
        if len(images) != len(masks):
            raise ValueError("The number of images and masks must be the same.")

        # 转换第一张图片为RGBA模式
        base_image_pil = tensor2pil(images[0]).convert('RGBA')
        
        for i in range(1, len(images)):
            # 获取当前图片并转换为RGBA
            current_image_pil = tensor2pil(images[i]).convert('RGBA')
            current_image_np = np.array(current_image_pil)
            
            # 处理mask
            mask_np = np.array(mask_to_pil(masks[i]))
            
            # 应用羽化效果
            if feather_amount > 0:
                mask_np = apply_feather(mask_np, feather_amount)
            
            # 创建完整的RGBA图像
            result_np = current_image_np.copy()
            result_np[..., 3] = mask_np  # 设置alpha通道
            
            # 转换回PIL图像
            masked_image = Image.fromarray(result_np, 'RGBA')
            
            # 合并到基础图像
            base_image_pil = Image.alpha_composite(base_image_pil, masked_image)

        # 转换回RGB并返回tensor
        combined_image_tensor = pil2tensor(base_image_pil.convert('RGB'))
        
        return (combined_image_tensor,)

NODE_CLASS_MAPPINGS.update({
    "MaskApplierAndCombiner": MaskApplierAndCombiner,
})

NODE_DISPLAY_NAME_MAPPINGS.update({
    "MaskApplierAndCombiner": "🎭Mask Applier and Combiner"
})


# From mask-processor-fixed.py
class MaskProcessor:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }
    
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("processed_mask",)
    FUNCTION = "process_mask"
    CATEGORY = "Image Processing"

    def is_valid_mask(self, mask):
        """
        检查mask是否有效
        只要mask不是全0或全1就认为是有效的
        """
        # 将tensor转换为numpy数组以便处理
        mask_np = mask.cpu().numpy()
        
        # 打印调试信息
        min_val = np.min(mask_np)
        max_val = np.max(mask_np)
        print(f"Mask value range: min={min_val}, max={max_val}")
        
        # 获取唯一值（使用更小的精度以处理浮点误差）
        unique_values = np.unique(np.round(mask_np, decimals=3))
        print(f"Unique values in mask: {unique_values}")
        
        # 只要mask不是全0或全1就认为是有效的
        is_all_zero = np.allclose(mask_np, 0, atol=0.1)
        is_all_one = np.allclose(mask_np, 1, atol=0.1)
        
        is_valid = not (is_all_zero or is_all_one)
        print(f"Is all zero: {is_all_zero}, Is all one: {is_all_one}, Is valid: {is_valid}")
        
        return is_valid

    def process_mask(self, image, mask=None):
        """
        处理输入图像和mask
        如果mask为空或无效，创建与输入图像相同尺寸的全1 mask
        如果mask不为空且有效，直接输出输入的mask
        """
        # 获取图像的维度信息
        batch_size, height, width, channels = image.shape
        
        # 检查mask是否为空或无效
        create_new_mask = True
        if mask is not None:
            # 验证mask维度
            if len(mask.shape) == 3 and mask.shape[0] == batch_size and mask.shape[1:] == (height, width):
                # 检查mask是否有效
                if self.is_valid_mask(mask):
                    processed_mask = mask
                    create_new_mask = False
                    print("Using existing valid mask")
                else:
                    print("Mask validation failed, creating new mask")
            else:
                print(f"Mask dimension mismatch. Expected {(batch_size, height, width)}, got {mask.shape}")
        else:
            print("No mask provided, creating new mask")

        if create_new_mask:
            # 创建新的全1 mask
            processed_mask = torch.ones((batch_size, height, width),
                                    dtype=image.dtype,
                                    device=image.device)
            
        return (processed_mask,)

# 节点映射
NODE_CLASS_MAPPINGS.update({
    "MaskProcessor": MaskProcessor
})

# 节点显示名称映射
NODE_DISPLAY_NAME_MAPPINGS.update({
    "MaskProcessor": "🎭Mask Processor"
})


# From node-merger.py
def merge_nodes(directory='.', output_file='merged_nodes.py'):
    # Store all node class names to avoid conflicts
    node_classes = set()
    
    # Store the merged content
    merged_content = [
        "# Merged ComfyUI Custom Nodes",
        "# Auto-generated file",
        "\nimport torch",
        "import numpy as np",
        "\n"
    ]
    
    # Track imports
    imports = set()
    
    def process_file(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Extract imports
        import_lines = re.findall(r'^(?:from|import).*$', content, re.MULTILINE)
        imports.update(import_lines)
        
        # Remove imports from content
        content = re.sub(r'^(?:from|import).*$\n?', '', content, flags=re.MULTILINE)
        
        # Extract class definitions
        class_matches = re.findall(r'class\s+(\w+)[\s\(]', content)
        node_classes.update(class_matches)
        
        return content.strip()
    
    # Process all .py files except the output file
    python_files = [f for f in os.listdir(directory) 
                if f.endswith('.py') 
                and f != output_file 
                and f != '__init__.py']
    
    # Add collected imports
    merged_content.extend(sorted(list(imports)))
    merged_content.append("\n")
    
    # Process and add each file's content
    for file in python_files:
        filepath = os.path.join(directory, file)
        file_content = process_file(filepath)
        merged_content.extend([
            f"\n# From {file}",
            file_content,
            "\n"
        ])
    
    # Write merged content
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(merged_content))
    
    # Generate __init__.py
    init_content = [
        "from .merged_nodes import *",
        "\n# Node class mapping",
        "NODE_CLASS_MAPPINGS = {",
    ]
    
    for node_class in sorted(node_classes):
        init_content.append(f"    '{node_class}': {node_class},")
    
    init_content.extend([
        "}",
        "\n# Node display name mapping",
        "NODE_DISPLAY_NAME_MAPPINGS = {",
    ])
    
    for node_class in sorted(node_classes):
        # Convert CamelCase to "Spaced Words"
        display_name = re.sub(r'(?<!^)(?=[A-Z])', ' ', node_class)
        init_content.append(f"    '{node_class}': '{display_name}',")
    
    init_content.append("}")
    
    with open('__init__.py', 'w', encoding='utf-8') as f:
        f.write('\n'.join(init_content))

if __name__ == '__main__':
    merge_nodes()
    print("Files merged successfully!")
    print("Created: merged_nodes.py and __init__.py")


# From tag-array-to-lines-node.py
class TagArrayToLines:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tag_array": ("STRING", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "convert_to_lines"
    CATEGORY = "text"

    def convert_to_lines(self, tag_array: str) -> tuple[str]:
        # Remove the surrounding brackets and replace ", " with newline
        tag_array = tag_array.strip('[]')
        result = tag_array.replace('", "', '\n')
        
        return (result,)

NODE_CLASS_MAPPINGS.update({
    "TagArrayToLines": TagArrayToLines
})

NODE_DISPLAY_NAME_MAPPINGS.update({
    "TagArrayToLines": "📄 Tag Array to Lines"
})


# From tag-index-merger-node.py
class TagIndexMerger:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tag_strings": ("STRING", {"forceInput": True}),
                "index_string": ("STRING", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "merge_tags_and_indices"
    CATEGORY = "image"

    def merge_tags_and_indices(self, tag_strings: str, index_string: str) -> tuple[str]:
        # Split the tag_strings into a list, removing empty lines
        tag_list = [tag.strip() for tag in tag_strings.split('\n') if tag.strip()]
        
        # Split and clean the index string
        indices = [idx.strip() for idx in index_string.split(',') if idx.strip()]
        
        # Ensure we have the same number of indices and tag strings
        if len(indices) != len(tag_list):
            raise ValueError(f"Number of indices ({len(indices)}) does not match number of tag strings ({len(tag_list)})")
        
        result = []
        for idx, tags in zip(indices, tag_list):
            merged = f'"{idx}":"{tags}"'
            result.append(merged)
        
        final_string = ','.join(result)
        return (final_string,)

NODE_CLASS_MAPPINGS.update({
    "TagIndexMerger": TagIndexMerger
})

NODE_DISPLAY_NAME_MAPPINGS.update({
    "TagIndexMerger": "🏷️ Tag-Index Merger"
})


# From text-processor.py
class GrabberTagProcessor:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {}
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("processed_text",)
    FUNCTION = 'process_text'
    CATEGORY = 'Text'

    def process_text(self, text):
        # Step 1: Replace all spaces with half-width commas
        text = text.replace(" ", ",")
        
        # Step 2: Replace all underscores with spaces
        text = text.replace("_", " ")
        
        return (text,)

NODE_CLASS_MAPPINGS.update({
    "GrabberTagProcessor": GrabberTagProcessor
})

NODE_DISPLAY_NAME_MAPPINGS.update({
    "GrabberTagProcessor": "Grabber Tag Processor"
})


# From updated-image-resizer.py
def pil2tensor(image: Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def tensor2pil(t_image: torch.Tensor) -> Image:
    return Image.fromarray(np.clip(255.0 * t_image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

class ImageResizer:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
                "resize_method": (["lanczos", "nearest-exact", "bilinear", "bicubic", "hamming", "box"], {
                    "default": "lanczos"
                }),
            },
            "optional": {}
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("resized_image",)
    FUNCTION = 'resize_image'
    CATEGORY = 'Image Processing'

    def resize_image(self, image_a, image_b, resize_method):
        # 缩放方式映射
        resize_methods = {
            "nearest-exact": Image.NEAREST,
            "bilinear": Image.BILINEAR,
            "lanczos": Image.LANCZOS,
            "bicubic": Image.BICUBIC,
            "hamming": Image.HAMMING,
            "box": Image.BOX
        }
        
        # 获取选择的缩放方式
        resize_filter = resize_methods.get(resize_method, Image.LANCZOS)
        
        def process_single_image(i_a, i_b):
            # Convert tensors to PIL for processing
            pil_a = tensor2pil(i_a.unsqueeze(0) if i_a.dim() == 3 else i_a).convert('RGB')
            pil_b = tensor2pil(i_b.unsqueeze(0) if i_b.dim() == 3 else i_b).convert('RGB')
            
            # Get target size from image_b
            target_size = pil_b.size
            
            # Resize image_a to the target size using selected algorithm
            resized_pil_a = pil_a.resize(target_size, resize_filter)
            
            # Convert resized PIL image back to tensor
            return pil2tensor(resized_pil_a)

        # Handle different input types
        if isinstance(image_a, list) and isinstance(image_b, list):
            # Both inputs are lists
            ret_images = [process_single_image(a, b) for a, b in zip(image_a, image_b)]
        elif isinstance(image_a, torch.Tensor) and isinstance(image_b, torch.Tensor):
            # Both inputs are tensors (potentially batches)
            if image_a.dim() == 4 and image_b.dim() == 4:
                # Batch processing
                ret_images = [process_single_image(a, b) for a, b in zip(image_a, image_b)]
            elif image_a.dim() == 3 and image_b.dim() == 3:
                # Single image processing
                ret_images = [process_single_image(image_a, image_b)]
            else:
                raise ValueError("Incompatible tensor dimensions")
        else:
            raise ValueError("Incompatible input types")
        
        return (torch.cat(ret_images, dim=0),)

NODE_CLASS_MAPPINGS.update({
    "ImageResizer": ImageResizer,
})

NODE_DISPLAY_NAME_MAPPINGS.update({
    "ImageResizer": "🖼️Image Resizer"
})


# From websocket_image_save.py
#You can use this node to save full size images through the websocket, the
#images will be sent in exactly the same format as the image previews: as
#binary images on the websocket with a 8 byte header indicating the type
#of binary message (first 4 bytes) and the image format (next 4 bytes).

#Note that no metadata will be put in the images saved with this node.

class SaveImageWebsocket:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"images": ("IMAGE", ),}
                }

    RETURN_TYPES = ()
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    CATEGORY = "api/image"

    def save_images(self, images):
        pbar = comfy.utils.ProgressBar(images.shape[0])
        step = 0
        for image in images:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            pbar.update_absolute(step, images.shape[0], ("PNG", img, None))
            step += 1

        return {}

    @classmethod
    def IS_CHANGED(s, images):
        return time.time()

NODE_CLASS_MAPPINGS.update({
    "SaveImageWebsocket": SaveImageWebsocket,
})


# From border_mask.py
class BorderMask:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "dist": ("INT", {
                    "default": 10,
                    "min": 0,
                    "max": 1000,
                    "step": 1
                }),
                "width": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 1000,
                    "step": 1
                })
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "generate_border_mask"
    CATEGORY = "Mask"

    def generate_border_mask(self, image, dist, width):
        # 获取图像尺寸
        batch_size, height, width_img, channels = image.shape
        
        # 创建全零mask
        mask = torch.zeros((batch_size, height, width_img), dtype=torch.float32, device=image.device)
        
        # 计算内边框的坐标
        inner_top = dist
        inner_bottom = height - dist
        inner_left = dist
        inner_right = width_img - dist
        
        # 计算外边框的坐标
        outer_top = dist + width
        outer_bottom = height - (dist + width)
        outer_left = dist + width
        outer_right = width_img - (dist + width)
        
        # 填充外边框区域为1
        mask[:, :outer_top, :] = 1  # 上边
        mask[:, outer_bottom:, :] = 1  # 下边
        mask[:, :, :outer_left] = 1  # 左边
        mask[:, :, outer_right:] = 1  # 右边
        
        # 填充内边框区域为0
        mask[:, inner_top:inner_bottom, inner_left:inner_right] = 0
        
        return (mask,)

NODE_CLASS_MAPPINGS.update({
    "BorderMask": BorderMask,
})

NODE_DISPLAY_NAME_MAPPINGS.update({
    "BorderMask": "🎭Border Mask"
})


# 增加新节点：SaturationAdjuster
def pil2tensor(image: Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def tensor2pil(t_image: torch.Tensor) -> Image:
    return Image.fromarray(np.clip(255.0 * t_image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

class SaturationAdjuster:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_image": ("IMAGE",),
                "luminance_image": ("IMAGE",),
                "effect_intensity": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.1
                }),
                "saturation_boost": ("FLOAT", {
                    "default": 1.5,
                    "min": 1.0,
                    "max": 3.0,
                    "step": 0.1
                }),
                "darkness_threshold": ("FLOAT", {
                    "default": 0.2,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "darkness_falloff": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.1
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("adjusted_image",)
    FUNCTION = 'adjust_saturation'
    CATEGORY = 'Image Processing'

    def adjust_saturation(self, base_image, luminance_image, effect_intensity, saturation_boost, darkness_threshold, darkness_falloff):
        # 处理单张图片的函数
        def process_single_image(base_img, lum_img):
            # 转换为PIL图像
            base_pil = tensor2pil(base_img).convert('RGB')
            
            # 将明度图调整为与底层图相同尺寸
            lum_pil = tensor2pil(lum_img).convert('L').resize(base_pil.size, Image.LANCZOS)
            
            # 获取明度图的数据
            lum_data = np.array(lum_pil)
            
            # 将底层图转换到HSV色彩空间
            base_hsv = np.array(base_pil.convert('HSV'))
            
            # 根据明度图的亮度值调整饱和度
            # 越暗的区域(明度图像素值低)，饱和度提升越多
            # 归一化明度值到0-1范围
            normalized_lum = lum_data / 255.0
            
            # 计算暗区减弱因子：当明度低于阈值时，根据减弱系数逐渐降低效果
            darkness_reduction = np.ones_like(normalized_lum)
            dark_mask = normalized_lum < darkness_threshold
            
            if np.any(dark_mask):
                # 将暗区的明度值映射到0-1范围内
                dark_values = normalized_lum[dark_mask] / darkness_threshold
                # 应用减弱曲线：值越低，减弱越明显
                dark_reduction = np.power(dark_values, darkness_falloff)
                darkness_reduction[dark_mask] = dark_reduction
            
            # 计算饱和度调整系数：越暗的区域系数越高，但应用暗区减弱
            adjustment = (1.0 - normalized_lum) * effect_intensity * darkness_reduction
            
            # 将调整应用到HSV图像的饱和度通道
            # HSV的S通道是1号通道
            saturation_channel = base_hsv[:, :, 1].astype(float)
            
            # 计算新的饱和度值
            new_saturation = saturation_channel * (1.0 + adjustment * (saturation_boost - 1.0))
            
            # 确保饱和度在0-255范围内
            new_saturation = np.clip(new_saturation, 0, 255)
            
            # 应用新的饱和度值
            base_hsv[:, :, 1] = new_saturation.astype(np.uint8)
            
            # 将HSV转换回RGB
            adjusted_pil = Image.fromarray(cv2.cvtColor(base_hsv, cv2.COLOR_HSV2RGB))
            
            # 转换回tensor
            return pil2tensor(adjusted_pil)
        
        # 处理输入图像
        batch_size = min(base_image.shape[0], luminance_image.shape[0])
        adjusted_images = []
        
        for i in range(batch_size):
            base_img = base_image[i] if base_image.dim() == 4 else base_image
            lum_img = luminance_image[i] if luminance_image.dim() == 4 else luminance_image
            
            adjusted = process_single_image(base_img, lum_img)
            adjusted_images.append(adjusted)
        
        # 合并所有处理后的图像
        return (torch.cat(adjusted_images, dim=0),)

# 更新节点映射
NODE_CLASS_MAPPINGS.update({
    "SaturationAdjuster": SaturationAdjuster,
})

NODE_DISPLAY_NAME_MAPPINGS.update({
    "SaturationAdjuster": "🌈SaturationAdjuster"
})


# 增加新节点：亮部覆盖 (Highlight Overlay)
def pil2tensor(image: Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def tensor2pil(t_image: torch.Tensor) -> Image:
    return Image.fromarray(np.clip(255.0 * t_image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

class HighlightOverlay:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
                "brightness_threshold": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "brightness_offset": ("FLOAT", {
                    "default": 0.0,
                    "min": -1.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "fill_holes": ("BOOLEAN", {
                    "default": True
                }),
                "hole_brightness_mode": (["周围平均亮度", "白色填充", "不改变"], {
                    "default": "周围平均亮度"
                }),
                "feather_amount": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100,
                    "step": 1
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("composite_image", "highlight_mask")
    FUNCTION = 'apply_highlight_overlay'
    CATEGORY = 'Image Processing'

    def calculate_luminance(self, rgb_image):
        """计算RGB图像的亮度（使用标准亮度公式）"""
        # 使用标准的亮度公式: Y = 0.299*R + 0.587*G + 0.114*B
        if isinstance(rgb_image, np.ndarray):
            if len(rgb_image.shape) == 3:
                r, g, b = rgb_image[:, :, 0], rgb_image[:, :, 1], rgb_image[:, :, 2]
                luminance = 0.299 * r + 0.587 * g + 0.114 * b
                return luminance
        return None

    def fill_mask_holes(self, mask):
        """填充mask中的空洞（封闭区域内的未填充部分）
        返回填充后的mask和空洞区域的mask"""
        # 将mask转换为二值图像（0和255）
        mask_binary = (mask * 255).astype(np.uint8)
        h, w = mask_binary.shape
        
        # 创建反转的mask（背景变前景，用于floodFill）
        inverted = 255 - mask_binary
        
        # 创建floodFill掩码（需要比原图大2像素）
        flood_mask = np.zeros((h + 2, w + 2), np.uint8)
        
        # 从四个角开始填充，标记所有与边缘相连的背景区域
        # 使用128作为标记值，区别于原始的255（空洞）和0（前景）
        corners = [(0, 0), (w-1, 0), (0, h-1), (w-1, h-1)]
        for x, y in corners:
            if inverted[y, x] == 255:  # 如果是背景像素
                cv2.floodFill(inverted, flood_mask, (x, y), 128)
        
        # 填充边缘的其他点（每隔一定距离采样）
        step = max(1, min(w, h) // 20)
        for x in range(0, w, step):
            if inverted[0, x] == 255:
                cv2.floodFill(inverted, flood_mask, (x, 0), 128)
            if h > 1 and inverted[h-1, x] == 255:
                cv2.floodFill(inverted, flood_mask, (x, h-1), 128)
        
        for y in range(0, h, step):
            if inverted[y, 0] == 255:
                cv2.floodFill(inverted, flood_mask, (0, y), 128)
            if w > 1 and inverted[y, w-1] == 255:
                cv2.floodFill(inverted, flood_mask, (w-1, y), 128)
        
        # 所有仍为255的区域就是封闭区域内的空洞（未与边缘相连的背景）
        holes = (inverted == 255).astype(np.float32)
        
        # 将空洞填充到原始mask中
        filled_mask = np.maximum(mask, holes)
        
        return filled_mask, holes

    def apply_feather_to_mask(self, mask, feather_amount):
        """对mask应用羽化效果"""
        if feather_amount <= 0:
            return mask
        
        # 将mask转换为0-255范围的uint8
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        # 使用高斯模糊实现羽化
        kernel_size = feather_amount * 2 + 1  # 确保核大小为奇数
        sigma = feather_amount / 2
        feathered = cv2.GaussianBlur(mask_uint8, (kernel_size, kernel_size), sigma)
        
        # 转换回0-1范围的float32
        feathered_mask = feathered.astype(np.float32) / 255.0
        
        return feathered_mask

    def calculate_surrounding_brightness(self, hsv_image, holes_mask, original_mask):
        """计算空洞周围区域的平均亮度"""
        # 找到空洞周围的区域（原始mask中为1但不在空洞内的区域）
        surrounding_mask = (original_mask > 0.5) & (holes_mask < 0.5)
        
        if not np.any(surrounding_mask):
            # 如果没有周围区域，使用整个mask区域的平均值
            surrounding_mask = original_mask > 0.5
        
        # 获取周围区域的V通道值
        v_channel = hsv_image[:, :, 2] / 255.0  # 归一化到0-1
        surrounding_values = v_channel[surrounding_mask]
        
        if len(surrounding_values) > 0:
            avg_brightness = np.mean(surrounding_values)
        else:
            avg_brightness = 0.5  # 默认值
        
        return avg_brightness

    def apply_highlight_overlay(self, image_a, image_b, brightness_threshold, brightness_offset, fill_holes, hole_brightness_mode, feather_amount):
        # 处理单张图片的函数
        def process_single_image(img_a, img_b):
            # 转换为PIL图像
            pil_a = tensor2pil(img_a).convert('RGB')
            pil_b = tensor2pil(img_b).convert('RGB')
            
            # 将 image_b 调整到 image_a 的尺寸
            pil_b_resized = pil_b.resize(pil_a.size, Image.LANCZOS)
            
            # 转换为numpy数组进行处理
            np_a = np.array(pil_a).astype(np.float32)
            np_b = np.array(pil_b_resized).astype(np.float32)
            
            # 计算两个图像的亮度
            lum_a = self.calculate_luminance(np_a)
            lum_b = self.calculate_luminance(np_b)
            
            # 归一化到0-1范围
            lum_a_norm = lum_a / 255.0
            lum_b_norm = lum_b / 255.0
            
            # 计算亮度差
            brightness_diff = lum_b_norm - lum_a_norm
            
            # 创建mask：找到 image_b 中亮度大于 image_a 的部分（考虑阈值）
            # mask值为1表示需要调整的区域
            mask = (brightness_diff > brightness_threshold).astype(np.float32)
            original_mask = mask.copy()  # 保存原始mask用于计算周围亮度
            
            # 将 image_a 转换到 HSV 色彩空间，以便只调整亮度（V通道）
            hsv_a = cv2.cvtColor(np_a.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv_b = cv2.cvtColor(np_b.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
            
            # 如果启用填充空洞，填充封闭区域内的空洞
            holes_mask = None
            if fill_holes:
                mask, holes_mask = self.fill_mask_holes(mask)
            
            # 保存填充后的mask用于后续处理（在羽化前）
            mask_before_feather = mask.copy()
            
            # 应用羽化效果
            if feather_amount > 0:
                mask = self.apply_feather_to_mask(mask, feather_amount)
            
            # 创建输出HSV图像，初始为 image_a
            hsv_result = hsv_a.copy()
            
            # 在mask区域，将 image_a 的亮度（V通道）调整到与 image_b 一致
            # V通道是第2个通道（索引为2）
            v_a = hsv_a[:, :, 2] / 255.0  # 归一化到0-1
            v_b = hsv_b[:, :, 2] / 255.0  # 归一化到0-1
            
            # 计算目标亮度值：image_b的亮度 + 偏移
            target_brightness = v_b + brightness_offset
            target_brightness = np.clip(target_brightness, 0.0, 1.0)  # 限制在0-1范围
            
            # 在mask区域应用亮度调整（排除空洞区域，空洞区域单独处理）
            v_result = v_a.copy()
            if holes_mask is not None and np.any(holes_mask > 0.5):
                # 分离空洞区域和普通mask区域（使用羽化前的mask进行分离）
                normal_mask = (mask_before_feather > 0.5) & (holes_mask < 0.5)
                holes_only = holes_mask > 0.5
                
                # 计算普通mask区域的混合权重（使用羽化后的mask值）
                normal_weight = np.where(normal_mask, mask, 0.0)
                # 计算空洞区域的混合权重（使用羽化后的mask值）
                holes_weight = np.where(holes_only, mask, 0.0)
                
                # 处理普通mask区域（使用羽化后的权重进行混合）
                v_result = v_result * (1.0 - normal_weight) + target_brightness * normal_weight
                
                # 处理空洞区域
                if hole_brightness_mode == "周围平均亮度":
                    # 计算周围区域的平均亮度
                    avg_brightness = self.calculate_surrounding_brightness(hsv_a, holes_mask, original_mask)
                    v_result = v_result * (1.0 - holes_weight) + avg_brightness * holes_weight
                elif hole_brightness_mode == "白色填充":
                    # 使用白色（最大亮度）
                    v_result = v_result * (1.0 - holes_weight) + 1.0 * holes_weight
                # "不改变" 模式不需要处理，保持原图亮度（holes_weight为0时不改变）
            else:
                # 没有空洞，直接应用亮度调整（使用羽化后的mask值进行混合）
                v_result = v_result * (1.0 - mask) + target_brightness * mask
            
            # 将亮度值转换回0-255范围并应用到HSV图像
            hsv_result[:, :, 2] = v_result * 255.0
            
            # 转换回RGB
            result_rgb = cv2.cvtColor(hsv_result.astype(np.uint8), cv2.COLOR_HSV2RGB)
            
            # 确保值在有效范围内
            result_rgb = np.clip(result_rgb, 0, 255).astype(np.uint8)
            
            # 转换回PIL图像
            result_pil = Image.fromarray(result_rgb)
            
            # 转换mask为tensor格式（0-1范围）
            mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()
            
            return pil2tensor(result_pil), mask_tensor
        
        # 处理输入图像
        batch_size = min(image_a.shape[0], image_b.shape[0])
        composite_images = []
        highlight_masks = []
        
        for i in range(batch_size):
            img_a = image_a[i] if image_a.dim() == 4 else image_a
            img_b = image_b[i] if image_b.dim() == 4 else image_b
            
            composite, mask = process_single_image(img_a, img_b)
            composite_images.append(composite)
            highlight_masks.append(mask)
        
        # 合并所有处理后的图像和mask
        composite_result = torch.cat(composite_images, dim=0)
        mask_result = torch.cat(highlight_masks, dim=0)
        
        return (composite_result, mask_result)

# 更新节点映射
NODE_CLASS_MAPPINGS.update({
    "HighlightOverlay": HighlightOverlay,
})

NODE_DISPLAY_NAME_MAPPINGS.update({
    "HighlightOverlay": "✨亮部覆盖 (Highlight Overlay)"
})


# Mask Merger Node - 合并多个mask
class MaskMerger:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "mask1": ("MASK",),
                "mask2": ("MASK",),
                "mask3": ("MASK",),
                "mask4": ("MASK",),
                "mask5": ("MASK",),
                "mask6": ("MASK",),
                "mask7": ("MASK",),
                "mask8": ("MASK",),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("merged_mask",)
    FUNCTION = "merge_masks"
    CATEGORY = "Mask"

    def is_empty_mask(self, mask):
        """检查mask是否为空"""
        if mask is None:
            return True
        if isinstance(mask, torch.Tensor):
            return mask.numel() == 0
        return False

    def merge_masks(self, mask1=None, mask2=None, mask3=None, mask4=None,
                    mask5=None, mask6=None, mask7=None, mask8=None):
        """
        合并多个mask，将蒙版区域累加
        - 忽略空的mask
        - 如果所有mask都为空，返回空mask
        """
        masks = [mask1, mask2, mask3, mask4, mask5, mask6, mask7, mask8]

        # 过滤出有效的mask
        valid_masks = [m for m in masks if not self.is_empty_mask(m)]

        # 如果没有有效的mask，返回空mask
        if not valid_masks:
            return (torch.empty((0, 0, 0), dtype=torch.float32),)

        # 使用第一个有效mask作为基准，获取尺寸和设备
        first_mask = valid_masks[0]
        device = first_mask.device
        dtype = first_mask.dtype

        # 获取目标尺寸（使用第一个mask的尺寸）
        if first_mask.dim() == 2:
            target_height, target_width = first_mask.shape
            target_shape = (target_height, target_width)
        else:  # dim == 3 (batch, height, width)
            target_shape = first_mask.shape[1:]

        # 初始化合并后的mask为零
        merged = torch.zeros(target_shape, dtype=dtype, device=device)

        # 累加所有有效的mask（使用最大值进行合并，避免超过1）
        for mask in valid_masks:
            # 处理不同维度的mask
            if mask.dim() == 2:
                # 单个mask，直接使用
                mask_resized = mask
            elif mask.dim() == 3:
                # batch mask，取第一个（或合并batch）
                # 如果batch size > 1，取batch中的最大值
                if mask.shape[0] > 1:
                    mask_resized = mask.max(dim=0)[0]
                else:
                    mask_resized = mask.squeeze(0)
            else:
                continue

            # 如果尺寸不匹配，进行调整
            if mask_resized.shape != target_shape:
                # 转换为numpy进行resize
                mask_np = mask_resized.cpu().numpy()
                mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8))
                mask_pil_resized = mask_pil.resize((target_shape[1], target_shape[0]), Image.NEAREST)
                mask_resized = torch.from_numpy(np.array(mask_pil_resized).astype(np.float32) / 255.0).to(device)

            # 使用最大值合并（这样重叠区域不会超过1）
            merged = torch.maximum(merged, mask_resized)

        # 添加batch维度
        merged = merged.unsqueeze(0)

        return (merged,)

NODE_CLASS_MAPPINGS.update({
    "MaskMerger": MaskMerger,
})

NODE_DISPLAY_NAME_MAPPINGS.update({
    "MaskMerger": "🎭Mask Merger"
})


# ==============================================================================
# Face Analysis Nodes (DeepFace)
# ==============================================================================

class FaceAnalysisNode:
    """
    Main node for facial expression analysis using DeepFace.
    Analyzes faces in images and returns emotion, age, gender, and race information.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "analyze_emotion": ("BOOLEAN", {"default": True}),
                "analyze_age": ("BOOLEAN", {"default": True}),
                "analyze_gender": ("BOOLEAN", {"default": True}),
                "analyze_race": ("BOOLEAN", {"default": False}),
                "detector_backend": (["retinaface", "mtcnn", "opencv", "ssd", "dlib", "mediapipe"], {"default": "retinaface"}),
                "enforce_detection": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("ANALYSIS_RESULT", "FACE_IMAGES", "IMAGE")
    RETURN_NAMES = ("analysis", "face_images", "visualization")
    FUNCTION = "analyze"
    CATEGORY = "Face Analysis"
    OUTPUT_NODE = True

    def analyze(
        self,
        image: torch.Tensor,
        analyze_emotion: bool = True,
        analyze_age: bool = True,
        analyze_gender: bool = True,
        analyze_race: bool = False,
        detector_backend: str = "retinaface",
        enforce_detection: bool = False,
    ):
        """
        Analyze faces in the input image.
        """
        from deepface import DeepFace

        # Handle batch dimension
        if len(image.shape) == 4:
            image_np = image[0].cpu().numpy()
        else:
            image_np = image.cpu().numpy()

        # Convert from RGB (0-1) to uint8
        image_np = (image_np * 255).astype(np.uint8)

        # Build actions list
        actions = []
        if analyze_emotion:
            actions.append("emotion")
        if analyze_age:
            actions.append("age")
        if analyze_gender:
            actions.append("gender")
        if analyze_race:
            actions.append("race")

        if not actions:
            actions = ["emotion"]

        try:
            result = DeepFace.analyze(
                img_path=image_np,
                actions=actions,
                enforce_detection=enforce_detection,
                detector_backend=detector_backend,
                align=True,
                silent=True,
            )

            if not isinstance(result, list):
                result = [result]

            analysis_results = []
            face_images = []
            visualization = image_np.copy()

            for idx, face in enumerate(result):
                region = face.get("region", {})
                x, y, w, h = region.get("x", 0), region.get("y", 0), region.get("w", 0), region.get("h", 0)

                if w > 0 and h > 0:
                    face_crop = image_np[y:y+h, x:x+w]
                    face_tensor = torch.from_numpy(face_crop.astype(np.float32) / 255.0)
                    face_images.append(face_tensor)

                    cv2.rectangle(visualization, (x, y), (x+w, y+h), (0, 255, 0), 2)

                    if analyze_emotion:
                        emotion = face.get("dominant_emotion", "")
                        cv2.putText(visualization, emotion, (x, y-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                face_result = {
                    "face_id": idx + 1,
                    "region": region,
                }

                if analyze_emotion:
                    face_result["emotion"] = face.get("dominant_emotion")
                    face_result["emotion_scores"] = face.get("emotion", {})

                if analyze_age:
                    face_result["age"] = face.get("age")

                if analyze_gender:
                    face_result["gender"] = face.get("dominant_gender")
                    face_result["gender_scores"] = face.get("gender", {})

                if analyze_race:
                    face_result["race"] = face.get("dominant_race")
                    face_result["race_scores"] = face.get("race", {})

                analysis_results.append(face_result)

            visualization_tensor = torch.from_numpy(visualization.astype(np.float32) / 255.0)

            return {
                "faces": analysis_results,
                "total_faces": len(analysis_results),
                "actions": actions,
            }, face_images, visualization_tensor

        except Exception as e:
            return {
                "faces": [],
                "total_faces": 0,
                "error": str(e),
            }, [], torch.from_numpy(image_np.astype(np.float32) / 255.0)


class EmotionAnalysisNode:
    """
    Node for detailed emotion analysis with description generation.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "detector_backend": (["retinaface", "mtcnn", "opencv", "ssd", "dlib", "mediapipe"], {"default": "retinaface"}),
                "generate_description": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING", "DICT", "IMAGE")
    RETURN_NAMES = ("emotion_text", "emotion_data", "visualization")
    FUNCTION = "analyze_emotion"
    CATEGORY = "Face Analysis"
    OUTPUT_NODE = True

    def analyze_emotion(
        self,
        image: torch.Tensor,
        detector_backend: str = "retinaface",
        generate_description: bool = True,
    ):
        """
        Analyze emotions and generate descriptions.
        """
        from deepface import DeepFace

        if len(image.shape) == 4:
            image_np = image[0].cpu().numpy()
        else:
            image_np = image.cpu().numpy()

        image_np = (image_np * 255).astype(np.uint8)

        emotion_descriptions = {
            'happy': 'appears happy and pleased',
            'sad': 'shows signs of sadness',
            'angry': 'appears angry or frustrated',
            'fear': 'shows signs of fear or concern',
            'surprise': 'appears surprised or amazed',
            'disgust': 'shows signs of disgust',
            'neutral': 'has a neutral expression'
        }

        try:
            result = DeepFace.analyze(
                img_path=image_np,
                actions=["emotion"],
                enforce_detection=False,
                detector_backend=detector_backend,
                align=True,
                silent=True,
            )

            if not isinstance(result, list):
                result = [result]

            emotion_texts = []
            all_emotion_data = []
            visualization = image_np.copy()

            for idx, face in enumerate(result):
                emotion = face.get('dominant_emotion', 'Unknown')
                emotion_scores = face.get('emotion', {})

                region = face.get("region", {})
                x, y, w, h = region.get("x", 0), region.get("y", 0), region.get("w", 0), region.get("h", 0)

                if w > 0 and h > 0:
                    cv2.rectangle(visualization, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(visualization, emotion, (x, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                if generate_description:
                    base_desc = emotion_descriptions.get(emotion, f'shows {emotion} expression')
                    text = f"Face {idx + 1}: {base_desc}. Dominant emotion: {emotion}"
                else:
                    text = f"Face {idx + 1}: {emotion}"

                emotion_texts.append(text)

                all_emotion_data.append({
                    "face_id": idx + 1,
                    "dominant_emotion": emotion,
                    "emotion_scores": {k: round(float(v), 2) for k, v in emotion_scores.items()} if emotion_scores else {}
                })

            visualization_tensor = torch.from_numpy(visualization.astype(np.float32) / 255.0)

            return "\n".join(emotion_texts), {"faces": all_emotion_data}, visualization_tensor

        except Exception as e:
            return f"Error: {str(e)}", {"error": str(e)}, torch.from_numpy(image_np.astype(np.float32) / 255.0)


class FaceExtractNode:
    """
    Node for extracting face regions from images.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "detector_backend": (["retinaface", "mtcnn", "opencv", "ssd", "dlib", "mediapipe"], {"default": "retinaface"}),
                "expand_ratio": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0}),
                "min_size": ("INT", {"default": 64, "min": 32, "max": 512}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("face_images", "face_masks", "face_count")
    FUNCTION = "extract_faces"
    CATEGORY = "Face Analysis"

    def extract_faces(
        self,
        image: torch.Tensor,
        detector_backend: str = "retinaface",
        expand_ratio: float = 0.2,
        min_size: int = 64,
    ):
        """
        Extract face regions from image.
        """
        from deepface import DeepFace

        if len(image.shape) == 4:
            images = image
        else:
            images = image.unsqueeze(0)

        all_faces = []
        all_masks = []

        for img in images:
            image_np = img.cpu().numpy()
            image_np = (image_np * 255).astype(np.uint8)
            h, w = image_np.shape[:2]

            try:
                faces = DeepFace.extract_faces(
                    img_path=image_np,
                    enforce_detection=False,
                    detector_backend=detector_backend,
                    align=True,
                )

                for face_data in faces:
                    region = face_data.get("facial_area", {})
                    x, y, fw, fh = region.get("x", 0), region.get("y", 0), region.get("w", 0), region.get("h", 0)

                    # Expand region
                    expand_w = int(fw * expand_ratio)
                    expand_h = int(fh * expand_ratio)
                    x1 = max(0, x - expand_w)
                    y1 = max(0, y - expand_h)
                    x2 = min(w, x + fw + expand_w)
                    y2 = min(h, y + fh + expand_h)

                    # Skip small faces
                    if (x2 - x1) < min_size or (y2 - y1) < min_size:
                        continue

                    face_crop = image_np[y1:y2, x1:x2]
                    face_tensor = torch.from_numpy(face_crop.astype(np.float32) / 255.0)
                    all_faces.append(face_tensor)

                    # Create mask
                    mask = np.zeros((h, w), dtype=np.float32)
                    mask[y1:y2, x1:x2] = 1.0
                    all_masks.append(torch.from_numpy(mask))

            except Exception:
                # Return empty if no faces found
                pass

        if all_faces:
            face_batch = torch.stack([f for f in all_faces])
            mask_batch = torch.stack([m for m in all_masks])
        else:
            face_batch = torch.empty((0, 3, min_size, min_size))
            mask_batch = torch.empty((0, min_size, min_size))

        return face_batch, mask_batch, str(len(all_faces))


class DescriptionGenNode:
    """
    Node for generating detailed facial expression descriptions.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "detector_backend": (["retinaface", "mtcnn", "opencv", "ssd", "dlib", "mediapipe"], {"default": "retinaface"}),
                "include_age": ("BOOLEAN", {"default": True}),
                "include_gender": ("BOOLEAN", {"default": True}),
                "include_race": ("BOOLEAN", {"default": False}),
                "language": (["english", "chinese"], {"default": "english"}),
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("description", "visualization")
    FUNCTION = "generate_description"
    CATEGORY = "Face Analysis"
    OUTPUT_NODE = True

    def generate_description(
        self,
        image: torch.Tensor,
        detector_backend: str = "retinaface",
        include_age: bool = True,
        include_gender: bool = True,
        include_race: bool = False,
        language: str = "english",
    ):
        """
        Generate detailed facial expression description.
        """
        from deepface import DeepFace

        if len(image.shape) == 4:
            image_np = image[0].cpu().numpy()
        else:
            image_np = image.cpu().numpy()

        image_np = (image_np * 255).astype(np.uint8)

        actions = ["emotion"]
        if include_age:
            actions.append("age")
        if include_gender:
            actions.append("gender")
        if include_race:
            actions.append("race")

        # Language-specific descriptions
        emotion_desc_en = {
            'happy': 'appears happy and pleased',
            'sad': 'shows signs of sadness',
            'angry': 'appears angry or frustrated',
            'fear': 'shows signs of fear or concern',
            'surprise': 'appears surprised or amazed',
            'disgust': 'shows signs of disgust',
            'neutral': 'has a neutral expression'
        }

        emotion_desc_cn = {
            'happy': '看起来开心愉快',
            'sad': '表现出悲伤的情绪',
            'angry': '看起来生气或沮丧',
            'fear': '表现出恐惧或担忧',
            'surprise': '看起来惊讶',
            'disgust': '表现出厌恶',
            'neutral': '表情平静'
        }

        emotion_desc = emotion_desc_cn if language == "chinese" else emotion_desc_en

        try:
            result = DeepFace.analyze(
                img_path=image_np,
                actions=actions,
                enforce_detection=False,
                detector_backend=detector_backend,
                align=True,
                silent=True,
            )

            if not isinstance(result, list):
                result = [result]

            descriptions = []
            visualization = image_np.copy()

            for idx, face in enumerate(result):
                region = face.get("region", {})
                x, y, w, h = region.get("x", 0), region.get("y", 0), region.get("w", 0), region.get("h", 0)

                if w > 0 and h > 0:
                    cv2.rectangle(visualization, (x, y), (x+w, y+h), (0, 255, 0), 2)

                parts = []

                # Emotion description
                emotion = face.get('dominant_emotion', 'Unknown')
                parts.append(emotion_desc.get(emotion, emotion))

                if include_age:
                    age = face.get('age')
                    if age:
                        if language == "chinese":
                            parts.append(f"年龄约{int(age)}岁")
                        else:
                            parts.append(f"age around {int(age)}")

                if include_gender:
                    gender = face.get('dominant_gender')
                    if gender:
                        if language == "chinese":
                            gender_cn = "男性" if gender == "Man" else "女性"
                            parts.append(gender_cn)
                        else:
                            parts.append(gender.lower())

                if include_race:
                    race = face.get('dominant_race')
                    if race:
                        parts.append(race)

                if language == "chinese":
                    desc = f"人脸{idx + 1}: " + "，".join(parts)
                else:
                    desc = f"Face {idx + 1}: " + ", ".join(parts)

                descriptions.append(desc)

                # Add text to visualization
                if w > 0 and h > 0:
                    emotion_short = emotion if language == "english" else emotion
                    cv2.putText(visualization, emotion_short, (x, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            visualization_tensor = torch.from_numpy(visualization.astype(np.float32) / 255.0)

            return "\n\n".join(descriptions), visualization_tensor

        except Exception as e:
            return f"Error: {str(e)}", torch.from_numpy(image_np.astype(np.float32) / 255.0)


NODE_CLASS_MAPPINGS.update({
    "FaceAnalysis": FaceAnalysisNode,
    "EmotionAnalysis": EmotionAnalysisNode,
    "FaceExtract": FaceExtractNode,
    "DescriptionGen": DescriptionGenNode,
})

NODE_DISPLAY_NAME_MAPPINGS.update({
    "FaceAnalysis": "😊 Face Analysis (DeepFace)",
    "EmotionAnalysis": "😊 Emotion Analysis",
    "FaceExtract": "😊 Face Extract",
    "DescriptionGen": "😊 Description Generator",
})


# ==============================================================================
# EmotiEffLib Face Analysis Nodes (Alternative - More Accurate)
# ==============================================================================

class EmotiEffLibAnalysisNode:
    """
    Emotion recognition using EmotiEffLib - More accurate than DeepFace.
    Uses EfficientNet models trained specifically for emotion recognition.
    """

    def __init__(self):
        self.recognizer = None
        self.face_detector = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "model_name": (["enet_b0_8_best_vgaf", "enet_b0_8_best_afew", "enet_b2_8", "enet_b0_8_va_mtl", "enet_b2_7"], {"default": "enet_b0_8_best_vgaf"}),
                "engine": (["onnx", "torch"], {"default": "onnx"}),
                "min_face_size": ("INT", {"default": 40, "min": 20, "max": 200}),
                "confidence_threshold": ("FLOAT", {"default": 0.9, "min": 0.5, "max": 1.0}),
            }
        }

    RETURN_TYPES = ("STRING", "DICT", "IMAGE")
    RETURN_NAMES = ("emotion_text", "emotion_data", "visualization")
    FUNCTION = "analyze_emotions"
    CATEGORY = "Face Analysis"
    OUTPUT_NODE = True

    def _init_detector(self, device):
        """Initialize face detector"""
        if self.face_detector is None:
            from facenet_pytorch import MTCNN
            self.face_detector = MTCNN(keep_all=True, post_process=False, min_face_size=40, device=device)
        return self.face_detector

    def _init_recognizer(self, engine, model_name, device):
        """Initialize emotion recognizer"""
        if self.recognizer is None:
            from emotiefflib.facial_analysis import EmotiEffLibRecognizer
            self.recognizer = EmotiEffLibRecognizer(engine=engine, model_name=model_name, device=device)
        return self.recognizer

    def analyze_emotions(
        self,
        image: torch.Tensor,
        model_name: str = "enet_b0_8_best_vgaf",
        engine: str = "onnx",
        min_face_size: int = 40,
        confidence_threshold: float = 0.9,
    ):
        """Analyze emotions using EmotiEffLib"""
        import torch as torch_module

        # Handle batch dimension
        if len(image.shape) == 4:
            image_np = image[0].cpu().numpy()
        else:
            image_np = image.cpu().numpy()

        # Convert from RGB (0-1) to uint8
        image_np = (image_np * 255).astype(np.uint8)

        # Determine device
        device = "cuda" if torch_module.cuda.is_available() else "cpu"

        try:
            # Initialize models
            detector = self._init_detector(device)
            recognizer = self._init_recognizer(engine, model_name, device)

            # Detect faces
            bounding_boxes, probs = detector.detect(image_np, landmarks=False)

            if bounding_boxes is None or len(bounding_boxes) == 0:
                return "No faces detected", {"faces": []}, image

            # Filter by confidence
            if probs is not None:
                valid_indices = probs > confidence_threshold
                bounding_boxes = bounding_boxes[valid_indices]

            emotion_texts = []
            all_emotion_data = []
            visualization = image_np.copy()

            h, w = image_np.shape[:2]

            for idx, bbox in enumerate(bounding_boxes):
                box = bbox.astype(int)
                x1, y1, x2, y2 = box[0:4]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                if x2 <= x1 or y2 <= y1:
                    continue

                # Crop face
                face_img = image_np[y1:y2, x1:x2]

                # Predict emotion
                emotion, scores = recognizer.predict_emotions(face_img, logits=False)

                # Get emotion labels
                emotion_labels = recognizer.idx_to_emotion_class

                # Build emotion scores dict
                emotion_scores = {emotion_labels[j]: float(scores[0][j]) for j in range(len(scores[0]))}

                # Draw on visualization
                cv2.rectangle(visualization, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(visualization, emotion[0], (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # Build text output
                text = f"Face {idx + 1}: {emotion[0]}"
                emotion_texts.append(text)

                all_emotion_data.append({
                    "face_id": idx + 1,
                    "dominant_emotion": emotion[0],
                    "emotion_scores": emotion_scores,
                    "bbox": [int(x1), int(y1), int(x2), int(y2)]
                })

            visualization_tensor = torch.from_numpy(visualization.astype(np.float32) / 255.0)

            return "\n".join(emotion_texts), {"faces": all_emotion_data}, visualization_tensor

        except ImportError as e:
            return f"Error: Missing dependency - {str(e)}", {"error": str(e)}, image
        except Exception as e:
            return f"Error: {str(e)}", {"error": str(e)}, image


class HSEmotionAnalysisNode:
    """
    Emotion recognition using HSEmotion library - High-speed and accurate.
    Alternative to DeepFace with better accuracy for emotion recognition.
    """

    def __init__(self):
        self.recognizer = None
        self.face_detector = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "model_name": (["enet_b0_8_best_afew", "enet_b0_8_best_vgaf", "enet_b0_8_va_mtl", "enet_b2_8", "enet_b2_7"], {"default": "enet_b0_8_best_afew"}),
                "min_face_size": ("INT", {"default": 40, "min": 20, "max": 200}),
            }
        }

    RETURN_TYPES = ("STRING", "DICT", "IMAGE")
    RETURN_NAMES = ("emotion_text", "emotion_data", "visualization")
    FUNCTION = "analyze_emotions"
    CATEGORY = "Face Analysis"
    OUTPUT_NODE = True

    def _init_detector(self, device):
        """Initialize face detector"""
        if self.face_detector is None:
            from facenet_pytorch import MTCNN
            self.face_detector = MTCNN(keep_all=True, post_process=False, min_face_size=40, device=device)
        return self.face_detector

    def _init_recognizer(self, model_name, device):
        """Initialize HSEmotion recognizer"""
        if self.recognizer is None:
            from hsemotion.facial_emotions import HSEmotionRecognizer
            self.recognizer = HSEmotionRecognizer(model_name=model_name, device=device)
        return self.recognizer

    def analyze_emotions(
        self,
        image: torch.Tensor,
        model_name: str = "enet_b0_8_best_afew",
        min_face_size: int = 40,
    ):
        """Analyze emotions using HSEmotion"""
        import torch as torch_module

        # Handle batch dimension
        if len(image.shape) == 4:
            image_np = image[0].cpu().numpy()
        else:
            image_np = image.cpu().numpy()

        # Convert from RGB (0-1) to uint8
        image_np = (image_np * 255).astype(np.uint8)

        # Determine device
        device = "gpu" if torch_module.cuda.is_available() else "cpu"

        try:
            # Initialize models
            detector = self._init_detector(device)
            recognizer = self._init_recognizer(model_name, device)

            # Detect faces
            bounding_boxes, probs = detector.detect(image_np, landmarks=False)

            if bounding_boxes is None or len(bounding_boxes) == 0:
                return "No faces detected", {"faces": []}, image

            # Filter by confidence
            if probs is not None:
                valid_indices = probs > 0.9
                bounding_boxes = bounding_boxes[valid_indices]

            emotion_texts = []
            all_emotion_data = []
            visualization = image_np.copy()

            h, w = image_np.shape[:2]

            for idx, bbox in enumerate(bounding_boxes):
                box = bbox.astype(int)
                x1, y1, x2, y2 = box[0:4]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                if x2 <= x1 or y2 <= y1:
                    continue

                # Crop face
                face_img = image_np[y1:y2, x1:x2]

                # Predict emotion
                emotion, scores = recognizer.predict_emotions(face_img, logits=True)

                # HSEmotion emotions: Anger, Contempt, Disgust, Fear, Happiness, Neutral, Sadness, Surprise
                emotion_labels = ["Anger", "Contempt", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise"]

                # Build emotion scores dict
                emotion_scores = {emotion_labels[j]: float(scores[j]) for j in range(min(len(scores), len(emotion_labels)))}

                # Draw on visualization
                cv2.rectangle(visualization, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(visualization, emotion, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # Build text output
                text = f"Face {idx + 1}: {emotion}"
                emotion_texts.append(text)

                all_emotion_data.append({
                    "face_id": idx + 1,
                    "dominant_emotion": emotion,
                    "emotion_scores": emotion_scores,
                    "bbox": [int(x1), int(y1), int(x2), int(y2)]
                })

            visualization_tensor = torch.from_numpy(visualization.astype(np.float32) / 255.0)

            return "\n".join(emotion_texts), {"faces": all_emotion_data}, visualization_tensor

        except ImportError as e:
            return f"Error: Missing dependency - {str(e)}\nInstall with: pip install hsemotion facenet-pytorch", {"error": str(e)}, image
        except Exception as e:
            return f"Error: {str(e)}", {"error": str(e)}, image


class MultiEngineEmotionNode:
    """
    Emotion recognition with multiple engine options.
    Compare results from DeepFace, EmotiEffLib, and HSEmotion.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "engine": (["deepface", "emotiefflib", "hsemotion"], {"default": "emotiefflib"}),
            },
            "optional": {
                "detector_backend": (["retinaface", "mtcnn", "opencv", "ssd"], {"default": "mtcnn"}),
            }
        }

    RETURN_TYPES = ("STRING", "DICT", "IMAGE")
    RETURN_NAMES = ("emotion_text", "emotion_data", "visualization")
    FUNCTION = "analyze_emotions"
    CATEGORY = "Face Analysis"
    OUTPUT_NODE = True

    def analyze_emotions(
        self,
        image: torch.Tensor,
        engine: str = "emotiefflib",
        detector_backend: str = "mtcnn",
    ):
        """Analyze emotions using selected engine"""

        # Handle batch dimension
        if len(image.shape) == 4:
            image_np = image[0].cpu().numpy()
        else:
            image_np = image.cpu().numpy()

        image_np = (image_np * 255).astype(np.uint8)

        if engine == "deepface":
            return self._analyze_deepface(image_np, image, detector_backend)
        elif engine == "emotiefflib":
            return self._analyze_emotiefflib(image_np, image)
        elif engine == "hsemotion":
            return self._analyze_hsemotion(image_np, image)
        else:
            return f"Unknown engine: {engine}", {"error": "Unknown engine"}, image

    def _analyze_deepface(self, image_np, original_image, detector_backend):
        """Use DeepFace for analysis"""
        try:
            from deepface import DeepFace

            result = DeepFace.analyze(
                img_path=image_np,
                actions=["emotion"],
                enforce_detection=False,
                detector_backend=detector_backend,
                align=True,
                silent=True,
            )

            if not isinstance(result, list):
                result = [result]

            emotion_texts = []
            all_emotion_data = []
            visualization = image_np.copy()

            for idx, face in enumerate(result):
                emotion = face.get('dominant_emotion', 'Unknown')
                emotion_scores = face.get('emotion', {})

                region = face.get("region", {})
                x, y, w, h = region.get("x", 0), region.get("y", 0), region.get("w", 0), region.get("h", 0)

                if w > 0 and h > 0:
                    cv2.rectangle(visualization, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(visualization, emotion, (x, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                emotion_texts.append(f"Face {idx + 1}: {emotion} (DeepFace)")
                all_emotion_data.append({
                    "face_id": idx + 1,
                    "engine": "deepface",
                    "dominant_emotion": emotion,
                    "emotion_scores": {k: round(float(v), 2) for k, v in emotion_scores.items()} if emotion_scores else {}
                })

            visualization_tensor = torch.from_numpy(visualization.astype(np.float32) / 255.0)
            return "\n".join(emotion_texts), {"faces": all_emotion_data, "engine": "deepface"}, visualization_tensor

        except Exception as e:
            return f"DeepFace Error: {str(e)}", {"error": str(e)}, original_image

    def _analyze_emotiefflib(self, image_np, original_image):
        """Use EmotiEffLib for analysis"""
        try:
            import torch as torch_module
            from facenet_pytorch import MTCNN
            from emotiefflib.facial_analysis import EmotiEffLibRecognizer

            device = "cuda" if torch_module.cuda.is_available() else "cpu"

            detector = MTCNN(keep_all=True, post_process=False, min_face_size=40, device=device)
            recognizer = EmotiEffLibRecognizer(engine="onnx", model_name="enet_b0_8_best_vgaf", device=device)

            bounding_boxes, probs = detector.detect(image_np, landmarks=False)

            if bounding_boxes is None or len(bounding_boxes) == 0:
                return "No faces detected", {"faces": []}, original_image

            if probs is not None:
                valid_indices = probs > 0.9
                bounding_boxes = bounding_boxes[valid_indices]

            emotion_texts = []
            all_emotion_data = []
            visualization = image_np.copy()
            h, w = image_np.shape[:2]

            for idx, bbox in enumerate(bounding_boxes):
                box = bbox.astype(int)
                x1, y1, x2, y2 = box[0:4]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                if x2 <= x1 or y2 <= y1:
                    continue

                face_img = image_np[y1:y2, x1:x2]
                emotion, scores = recognizer.predict_emotions(face_img, logits=False)
                emotion_labels = recognizer.idx_to_emotion_class
                emotion_scores = {emotion_labels[j]: float(scores[0][j]) for j in range(len(scores[0]))}

                cv2.rectangle(visualization, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(visualization, emotion[0], (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                emotion_texts.append(f"Face {idx + 1}: {emotion[0]} (EmotiEffLib)")
                all_emotion_data.append({
                    "face_id": idx + 1,
                    "engine": "emotiefflib",
                    "dominant_emotion": emotion[0],
                    "emotion_scores": emotion_scores
                })

            visualization_tensor = torch.from_numpy(visualization.astype(np.float32) / 255.0)
            return "\n".join(emotion_texts), {"faces": all_emotion_data, "engine": "emotiefflib"}, visualization_tensor

        except ImportError as e:
            return f"EmotiEffLib Error: Missing dependency - {str(e)}", {"error": str(e)}, original_image
        except Exception as e:
            return f"EmotiEffLib Error: {str(e)}", {"error": str(e)}, original_image

    def _analyze_hsemotion(self, image_np, original_image):
        """Use HSEmotion for analysis"""
        try:
            import torch as torch_module
            from facenet_pytorch import MTCNN
            from hsemotion.facial_emotions import HSEmotionRecognizer

            device = "gpu" if torch_module.cuda.is_available() else "cpu"

            detector = MTCNN(keep_all=True, post_process=False, min_face_size=40, device=device)
            recognizer = HSEmotionRecognizer(model_name="enet_b0_8_best_afew", device=device)

            bounding_boxes, probs = detector.detect(image_np, landmarks=False)

            if bounding_boxes is None or len(bounding_boxes) == 0:
                return "No faces detected", {"faces": []}, original_image

            if probs is not None:
                valid_indices = probs > 0.9
                bounding_boxes = bounding_boxes[valid_indices]

            emotion_texts = []
            all_emotion_data = []
            visualization = image_np.copy()
            h, w = image_np.shape[:2]
            emotion_labels = ["Anger", "Contempt", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise"]

            for idx, bbox in enumerate(bounding_boxes):
                box = bbox.astype(int)
                x1, y1, x2, y2 = box[0:4]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                if x2 <= x1 or y2 <= y1:
                    continue

                face_img = image_np[y1:y2, x1:x2]
                emotion, scores = recognizer.predict_emotions(face_img, logits=False)
                emotion_scores = {emotion_labels[j]: float(scores[j]) for j in range(min(len(scores), len(emotion_labels)))}

                cv2.rectangle(visualization, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(visualization, emotion, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                emotion_texts.append(f"Face {idx + 1}: {emotion} (HSEmotion)")
                all_emotion_data.append({
                    "face_id": idx + 1,
                    "engine": "hsemotion",
                    "dominant_emotion": emotion,
                    "emotion_scores": emotion_scores
                })

            visualization_tensor = torch.from_numpy(visualization.astype(np.float32) / 255.0)
            return "\n".join(emotion_texts), {"faces": all_emotion_data, "engine": "hsemotion"}, visualization_tensor

        except ImportError as e:
            return f"HSEmotion Error: Missing dependency - {str(e)}\nInstall with: pip install hsemotion facenet-pytorch", {"error": str(e)}, original_image
        except Exception as e:
            return f"HSEmotion Error: {str(e)}", {"error": str(e)}, original_image


NODE_CLASS_MAPPINGS.update({
    "EmotiEffLibAnalysis": EmotiEffLibAnalysisNode,
    "HSEmotionAnalysis": HSEmotionAnalysisNode,
    "MultiEngineEmotion": MultiEngineEmotionNode,
})

NODE_DISPLAY_NAME_MAPPINGS.update({
    "EmotiEffLibAnalysis": "😊 EmotiEffLib Analysis",
    "HSEmotionAnalysis": "😊 HSEmotion Analysis",
    "MultiEngineEmotion": "😊 Multi-Engine Emotion",
})

