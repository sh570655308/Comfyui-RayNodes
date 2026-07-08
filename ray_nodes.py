# Merged ComfyUI Custom Nodes
# Auto-generated file

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple  # 添加这行
from PIL import Image
import cv2
import os
import re
import time
import json
import comfy.utils
import comfy.sd
import folder_paths


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


# RTX+lanczos 缩放辅助函数
def rtx_upscale_image(image_tensor, target_w, target_h, quality="ULTRA"):
    """
    使用RTX VSR进行放大，然后用lanczos缩小到目标分辨率
    如果目标分辨率比原图大，直接用RTX放大
    如果目标分辨率比原图小，先用RTX放大到2x，然后用lanczos缩小
    """
    try:
        import nvvfx
    except ImportError:
        raise ImportError("nvvfx库未安装，无法使用RTX缩放。请安装NVIDIA RTX VSR。")

    orig_h, orig_w = image_tensor.shape[:2]

    # 确定RTX放大的目标尺寸
    # 如果目标比原图大，直接用RTX放大到目标
    # 如果目标比原图小，用RTX放大到2x，然后用lanczos缩小
    if target_w >= orig_w and target_h >= orig_h:
        # 目标更大，直接用RTX放大
        rtx_w, rtx_h = target_w, target_h
    else:
        # 目标更小，RTX放大2x后用lanczos缩小
        rtx_w = max(target_w, orig_w * 2)
        rtx_h = max(target_h, orig_h * 2)

    # 确保是8的倍数（RTX要求）
    rtx_w = max(8, round(rtx_w / 8) * 8)
    rtx_h = max(8, round(rtx_h / 8) * 8)

    # 质量映射
    quality_mapping = {
        "LOW": nvvfx.effects.QualityLevel.LOW,
        "MEDIUM": nvvfx.effects.QualityLevel.MEDIUM,
        "HIGH": nvvfx.effects.QualityLevel.HIGH,
        "ULTRA": nvvfx.effects.QualityLevel.ULTRA,
    }
    selected_quality = quality_mapping.get(quality, nvvfx.effects.QualityLevel.ULTRA)

    # RTX放大
    with nvvfx.VideoSuperRes(selected_quality) as sr:
        sr.output_width = rtx_w
        sr.output_height = rtx_h
        sr.load()

        # 准备输入 (HWC -> CHW)
        input_frame = image_tensor.cuda().permute(2, 0, 1).unsqueeze(0).contiguous()
        dlpack_out = sr.run(input_frame[0]).image
        rtx_output = torch.from_dlpack(dlpack_out).clone()

        # CHW -> HWC
        rtx_output = rtx_output.permute(1, 2, 0).cpu()

    # 如果RTX输出尺寸与目标不同，用lanczos调整
    if rtx_w != target_w or rtx_h != target_h:
        pil_img = Image.fromarray((rtx_output.numpy() * 255).astype(np.uint8))
        resized_pil = pil_img.resize((target_w, target_h), Image.LANCZOS)
        result = torch.from_numpy(np.array(resized_pil).astype(np.float32) / 255.0)
    else:
        result = rtx_output

    return result


class ImageResizer:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
                "resize_method": (["rtx+lanczos", "lanczos", "nearest-exact", "bilinear", "bicubic", "hamming", "box"], {
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
        # 缩放方式映射（不含rtx+lanczos）
        resize_methods = {
            "nearest-exact": Image.NEAREST,
            "bilinear": Image.BILINEAR,
            "lanczos": Image.LANCZOS,
            "bicubic": Image.BICUBIC,
            "hamming": Image.HAMMING,
            "box": Image.BOX
        }

        def process_single_image(i_a, i_b):
            # 获取目标尺寸
            target_h, target_w = i_b.shape[:2]

            if resize_method == "rtx+lanczos":
                # 使用RTX+lanczos
                return rtx_upscale_image(i_a, target_w, target_h, "ULTRA").unsqueeze(0)
            else:
                # 使用传统PIL方法
                resize_filter = resize_methods.get(resize_method, Image.LANCZOS)
                pil_a = tensor2pil(i_a.unsqueeze(0) if i_a.dim() == 3 else i_a).convert('RGB')
                resized_pil_a = pil_a.resize((target_w, target_h), resize_filter)
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


# ==============================================================================
# Facial Action Unit Detection Node (py-feat)
# ==============================================================================

# Fix scipy compatibility issue: simps was renamed to simpson in scipy 1.10+
try:
    from scipy.integrate import simps
except ImportError:
    from scipy.integrate import simpson as simps
    import scipy.integrate
    scipy.integrate.simps = simps

# Action Unit names and descriptions
AU_DESCRIPTIONS = {
    "AU1": {"name": "Inner Brow Raiser", "cn": "内侧眉毛上扬", "muscle": "Frontalis (medial)"},
    "AU2": {"name": "Outer Brow Raiser", "cn": "外侧眉毛上扬", "muscle": "Frontalis (lateral)"},
    "AU4": {"name": "Brow Lowerer", "cn": "蹙眉", "muscle": "Corrugator Supercilii"},
    "AU5": {"name": "Upper Lid Raiser", "cn": "上眼睑上扬", "muscle": "Levator Palpebrae"},
    "AU6": {"name": "Cheek Raiser", "cn": "脸颊上扬", "muscle": "Orbicularis Oculi"},
    "AU7": {"name": "Lid Tightener", "cn": "眼睑紧绷", "muscle": "Orbicularis Oculi"},
    "AU9": {"name": "Nose Wrinkler", "cn": "皱鼻", "muscle": "Levator Labii Superioris"},
    "AU10": {"name": "Upper Lip Raiser", "cn": "上唇上扬", "muscle": "Levator Labii Superioris"},
    "AU11": {"name": "Nasolabial Deepener", "cn": "鼻唇沟加深", "muscle": "Zygomaticus Minor"},
    "AU12": {"name": "Lip Corner Puller", "cn": "嘴角上扬", "muscle": "Zygomaticus Major"},
    "AU14": {"name": "Dimpler", "cn": "酒窝", "muscle": "Buccinator"},
    "AU15": {"name": "Lip Corner Depressor", "cn": "嘴角下垂", "muscle": "Depressor Anguli Oris"},
    "AU17": {"name": "Chin Raiser", "cn": "下巴上扬", "muscle": "Mentalis"},
    "AU20": {"name": "Lip Stretcher", "cn": "嘴唇拉伸", "muscle": "Risorius"},
    "AU23": {"name": "Lip Tightener", "cn": "嘴唇紧闭", "muscle": "Orbicularis Oris"},
    "AU24": {"name": "Lip Pressor", "cn": "嘴唇抿紧", "muscle": "Orbicularis Oris"},
    "AU25": {"name": "Lips Part", "cn": "嘴唇张开", "muscle": "Depressor Labii"},
    "AU26": {"name": "Jaw Drop", "cn": "下巴下垂", "muscle": "Masseter"},
    "AU28": {"name": "Lip Suck", "cn": "嘴唇吸吮", "muscle": "Orbicularis Oris"},
    "AU43": {"name": "Eyes Closed", "cn": "闭眼", "muscle": "Relaxation"},
}


class ActionUnitDetectionNode:
    """
    Facial Action Unit (AU) Detection using py-feat with EmotiEffLib face detector.
    Detects specific facial muscle movements like "furrowed brows", "smiling", etc.
    Works better with anime/illustration faces than default py-feat.
    """

    def __init__(self):
        self.au_detector = None
        self.face_detector = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "au_model": (["xgb", "svm"], {"default": "xgb"}),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "output_format": (["description", "tags", "json"], {"default": "description"}),
                "language": (["english", "chinese"], {"default": "english"}),
                "include_landmarks": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING", "DICT", "IMAGE")
    RETURN_NAMES = ("au_text", "au_data", "visualization")
    FUNCTION = "detect_action_units"
    CATEGORY = "Face Analysis"
    OUTPUT_NODE = True

    def _init_face_detector(self, device):
        """Initialize MTCNN face detector from facenet-pytorch"""
        if self.face_detector is None:
            from facenet_pytorch import MTCNN
            self.face_detector = MTCNN(keep_all=True, post_process=False, min_face_size=20, device=device)
        return self.face_detector

    def _init_au_detector(self, au_model):
        """Initialize py-feat AU detector"""
        if self.au_detector is None:
            from feat import Detector
            # py-feat requires all models to be specified, cannot disable individual detections
            self.au_detector = Detector(
                au_model=au_model,
                face_model='retinaface',
            )
        return self.au_detector

    def detect_action_units(
        self,
        image: torch.Tensor,
        au_model: str = "xgb",
        threshold: float = 0.5,
        output_format: str = "description",
        language: str = "english",
        include_landmarks: bool = False,
    ):
        """Detect facial action units using MTCNN for face detection and py-feat for AU analysis"""
        import tempfile
        import os
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

        temp_file = None
        try:
            # Step 1: Use MTCNN to detect faces (better for anime)
            face_detector = self._init_face_detector(device)
            detect_result = face_detector.detect(image_np, landmarks=True)
            # MTCNN returns (boxes, probs, landmarks) when landmarks=True
            if len(detect_result) == 3:
                bounding_boxes, probs, landmarks = detect_result
            else:
                bounding_boxes, probs = detect_result
                landmarks = None

            if bounding_boxes is None or len(bounding_boxes) == 0:
                return "No faces detected", {"faces": [], "error": "No faces detected"}, image

            # Filter by confidence
            if probs is not None:
                valid_indices = probs > 0.7
                bounding_boxes = bounding_boxes[valid_indices]

            if len(bounding_boxes) == 0:
                return "No faces detected (low confidence)", {"faces": [], "error": "No faces detected"}, image

            # Step 2: Save image and use py-feat for AU detection
            temp_fd, temp_path = tempfile.mkstemp(suffix='.png')
            temp_file = temp_path
            os.close(temp_fd)

            pil_img = Image.fromarray(image_np)
            pil_img.save(temp_path)

            au_detector = self._init_au_detector(au_model)
            result = au_detector.detect_image(temp_path)

            # Check if AU detection worked
            if result.aus is None or len(result.aus) == 0:
                return "AU detection failed", {"faces": [], "error": "AU detection failed"}, image

            au_texts = []
            all_au_data = []
            visualization = image_np.copy()

            # Get AU columns
            au_columns = [col for col in result.aus.columns if col.startswith('AU')]
            num_faces = min(len(bounding_boxes), len(result.aus))

            h, w = image_np.shape[:2]

            for face_idx in range(num_faces):
                # Get bounding box from MTCNN
                bbox = bounding_boxes[face_idx]
                box = bbox.astype(int)
                x1, y1, x2, y2 = max(0, box[0]), max(0, box[1]), min(w, box[2]), min(h, box[3])

                # Get AU data
                face_row = result.aus.iloc[face_idx]

                # Collect detected AUs above threshold
                detected_aus = []
                au_scores = {}

                for au_col in au_columns:
                    au_num = au_col.replace('AU', '')
                    au_name = f"AU{int(au_num)}"
                    score = float(face_row[au_col])
                    au_scores[au_name] = score

                    if score >= threshold:
                        detected_aus.append({
                            "au": au_name,
                            "score": score,
                            "description": AU_DESCRIPTIONS.get(au_name, {"name": au_name, "cn": au_name})
                        })

                # Sort by score
                detected_aus.sort(key=lambda x: x["score"], reverse=True)

                # Build output text - simplified format
                if output_format == "tags":
                    if language == "chinese":
                        tags = [AU_DESCRIPTIONS.get(au["au"], {"cn": au["au"]})["cn"] for au in detected_aus]
                    else:
                        tags = [AU_DESCRIPTIONS.get(au["au"], {"name": au["au"]})["name"] for au in detected_aus]
                    text = ", ".join(tags)
                elif output_format == "json":
                    import json
                    text = json.dumps({
                        "face_id": face_idx + 1,
                        "detected_aus": [{"au": au["au"], "score": round(au["score"], 3)} for au in detected_aus]
                    }, indent=2)
                else:
                    # Description format - simple, no AU numbers or confidence
                    lines = [f"Face {face_idx + 1}:"]
                    for au in detected_aus:
                        desc = AU_DESCRIPTIONS.get(au["au"], {"name": au["au"], "cn": au["au"]})
                        if language == "chinese":
                            lines.append(f"  - {desc['cn']}")
                        else:
                            lines.append(f"  - {desc['name']}")
                    text = "\n".join(lines)

                au_texts.append(text)

                # Store data
                face_data = {
                    "face_id": face_idx + 1,
                    "detected_aus": [{"au": au["au"], "score": round(au["score"], 3)} for au in detected_aus],
                    "all_au_scores": {k: round(v, 3) for k, v in au_scores.items()},
                }

                all_au_data.append(face_data)

                # Draw on visualization
                if x2 > x1 and y2 > y1:
                    cv2.rectangle(visualization, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    if detected_aus:
                        top_au = detected_aus[0]
                        desc = AU_DESCRIPTIONS.get(top_au["au"], {"name": top_au["au"], "cn": top_au["au"]})
                        label = desc["cn"] if language == "chinese" else desc["name"]
                        cv2.putText(visualization, label, (x1, y1-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            visualization_tensor = torch.from_numpy(visualization.astype(np.float32) / 255.0)

            return "\n\n".join(au_texts), {"faces": all_au_data, "model": au_model, "threshold": threshold}, visualization_tensor

        except ImportError as e:
            return f"Error: Missing dependency - {str(e)}", {"error": str(e)}, image
        except Exception as e:
            import traceback
            return f"Error: {str(e)}\n{traceback.format_exc()}", {"error": str(e)}, image
        finally:
            if temp_file and os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass


class ActionUnitDescriptionNode:
    """
    Generate natural language descriptions from Action Units.
    Converts detected AUs into readable facial expression descriptions.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "au_data": ("DICT",),
            },
            "optional": {
                "language": (["english", "chinese"], {"default": "english"}),
                "detail_level": (["simple", "detailed", "comprehensive"], {"default": "detailed"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("description",)
    FUNCTION = "generate_description"
    CATEGORY = "Face Analysis"

    def generate_description(self, au_data: dict, language: str = "english", detail_level: str = "detailed"):
        """Generate natural language description from AU data"""
        if "error" in au_data:
            return (f"Error: {au_data['error']}",)

        faces = au_data.get("faces", [])
        if not faces:
            return ("No action unit data available",)

        descriptions = []

        # AU to expression mappings
        au_expressions_en = {
            "AU1": "raised inner eyebrows",
            "AU2": "raised outer eyebrows",
            "AU4": "furrowed brows",
            "AU5": "widened eyes",
            "AU6": "raised cheeks (smiling eyes)",
            "AU7": "tightened eyelids",
            "AU9": "wrinkled nose",
            "AU10": "raised upper lip",
            "AU11": "deepened nasolabial folds",
            "AU12": "smiling (mouth corners pulled up)",
            "AU13": "puffed cheeks",
            "AU14": "dimpling",
            "AU15": "mouth corners turned down",
            "AU17": "raised chin",
            "AU18": "puckered lips",
            "AU19": "tongue visible",
            "AU20": "stretched lips",
            "AU22": "funneled lips",
            "AU23": "tightened lips",
            "AU24": "pressed lips",
            "AU25": "parted lips",
            "AU26": "dropped jaw",
            "AU27": "mouth stretched open",
            "AU28": "sucked lip",
            "AU43": "closed eyes",
            "AU45": "blinking",
        }

        au_expressions_cn = {
            "AU1": "内侧眉毛上扬",
            "AU2": "外侧眉毛上扬",
            "AU4": "蹙眉",
            "AU5": "眼睛睁大",
            "AU6": "脸颊上扬（笑眼）",
            "AU7": "眼睑紧绷",
            "AU9": "皱鼻",
            "AU10": "上唇上扬",
            "AU11": "鼻唇沟加深",
            "AU12": "微笑（嘴角上扬）",
            "AU13": "脸颊鼓起",
            "AU14": "酒窝",
            "AU15": "嘴角下垂",
            "AU17": "下巴上扬",
            "AU18": "嘴唇嘟起",
            "AU19": "舌头露出",
            "AU20": "嘴唇拉伸",
            "AU22": "嘴唇嘟成圆形",
            "AU23": "嘴唇紧绷",
            "AU24": "嘴唇抿紧",
            "AU25": "嘴唇微张",
            "AU26": "下巴下垂（张嘴）",
            "AU27": "嘴巴张大",
            "AU28": "嘴唇吸吮",
            "AU43": "闭眼",
            "AU45": "眨眼",
        }

        au_expr = au_expressions_cn if language == "chinese" else au_expressions_en

        for face in faces:
            face_id = face.get("face_id", 1)
            detected_aus = face.get("detected_aus", [])

            if not detected_aus:
                if language == "chinese":
                    descriptions.append(f"人脸 {face_id}: 未检测到明显的面部动作")
                else:
                    descriptions.append(f"Face {face_id}: No significant facial actions detected")
                continue

            if detail_level == "simple":
                # Just list the top 3 actions
                top_aus = detected_aus[:3]
                actions = [au_expr.get(au["au"], au["au"]) for au in top_aus]
                if language == "chinese":
                    desc = f"人脸 {face_id}: " + "、".join(actions)
                else:
                    desc = f"Face {face_id}: " + ", ".join(actions)

            elif detail_level == "detailed":
                # List actions only (no confidence)
                lines = [f"Face {face_id}:" if language == "english" else f"人脸 {face_id}:"]
                for au in detected_aus[:5]:  # Top 5
                    action = au_expr.get(au["au"], au["au"])
                    lines.append(f"  - {action}")
                desc = "\n".join(lines)

            else:  # comprehensive
                # Full description with all actions (no AU names or confidence)
                lines = [f"Face {face_id} - Facial Actions:" if language == "english" else f"人脸 {face_id} - 面部动作:"]
                lines.append("")
                for au in detected_aus:
                    action = au_expr.get(au["au"], au["au"])
                    lines.append(f"• {action}")
                desc = "\n".join(lines)

            descriptions.append(desc)

        return ("\n\n".join(descriptions),)


NODE_CLASS_MAPPINGS.update({
    "ActionUnitDetection": ActionUnitDetectionNode,
    "ActionUnitDescription": ActionUnitDescriptionNode,
})

NODE_DISPLAY_NAME_MAPPINGS.update({
    "ActionUnitDetection": "😊 Action Unit Detection (FACS)",
    "ActionUnitDescription": "😊 Action Unit Description",
})


# 增加新节点：PixelCountScaler - 根据像素总数缩放图像，并支持倍率限制
class PixelCountScaler:
    """
    根据目标像素总数缩放图像，并将边长调整为指定倍数。
    像素总数使用二进制概念（如1M = 1024*1024 = 1,048,576）
    支持GPU批量加速和进度显示。
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "target_megapixels": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.001,
                    "max": 1000.0,
                    "step": 0.001,
                    "display": "number"
                }),
                "divisor": ("INT", {
                    "default": 8,
                    "min": 1,
                    "max": 128,
                    "step": 1
                }),
                "resize_method": (["rtx+lanczos", "lanczos", "nearest-exact", "bilinear", "bicubic", "hamming", "box"], {
                    "default": "lanczos"
                }),
            },
            "optional": {}
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("resized_image", "new_width", "new_height")
    FUNCTION = 'scale_by_pixel_count'
    CATEGORY = 'Image Processing'

    # torch支持的resize模式映射
    TORCH_RESIZE_MODES = {
        "nearest-exact": "nearest-exact",
        "bilinear": "bilinear",
        "bicubic": "bicubic",
        "box": "area",
    }

    def scale_by_pixel_count(self, image, target_megapixels, divisor, resize_method):
        # 计算实际目标像素数（1M = 1024×1024 = 1,048,576）
        total_target_pixels = int(target_megapixels * 1024 * 1024)
        total_target_pixels = max(1, total_target_pixels)

        # PIL缩放方式映射
        pil_resize_methods = {
            "nearest-exact": Image.NEAREST,
            "bilinear": Image.BILINEAR,
            "lanczos": Image.LANCZOS,
            "bicubic": Image.BICUBIC,
            "hamming": Image.HAMMING,
            "box": Image.BOX
        }

        # 计算目标尺寸（基于第一张图的比例）
        orig_h, orig_w = image.shape[1], image.shape[2]
        orig_pixels = orig_w * orig_h
        scale = (total_target_pixels / max(1, orig_pixels)) ** 0.5

        def round_to_divisor(value, d):
            return max(d, round(value / d) * d)

        final_w = round_to_divisor(orig_w * scale, divisor)
        final_h = round_to_divisor(orig_h * scale, divisor)

        # 单张图像：增加batch维度统一处理
        if image.dim() == 3:
            image = image.unsqueeze(0)

        batch_size = image.shape[0]

        if resize_method == "rtx+lanczos":
            resized_images = self._resize_rtx_batch(image, final_w, final_h, batch_size)
        elif resize_method in self.TORCH_RESIZE_MODES:
            # GPU批量resize（nearest-exact, bilinear, bicubic, box）
            resized_images = self._resize_torch_batch(image, final_w, final_h, resize_method)
        else:
            # PIL逐张处理（lanczos, hamming）带进度条
            resized_images = self._resize_pil_batch(image, final_w, final_h, pil_resize_methods[resize_method], batch_size)

        return (resized_images, final_w, final_h)

    def _resize_torch_batch(self, image, target_w, target_h, resize_method):
        """使用torch.nn.functional.interpolate在GPU上批量resize"""
        mode = self.TORCH_RESIZE_MODES[resize_method]
        # BHWC -> BCHW, 移到GPU
        tensor = image.permute(0, 3, 1, 2).cuda()
        # 批量resize
        resized = F.interpolate(tensor, size=(target_h, target_w), mode=mode)
        # BCHW -> BHWC, 回CPU
        return resized.permute(0, 2, 3, 1).cpu().clamp(0.0, 1.0)

    def _resize_pil_batch(self, image, target_w, target_h, pil_filter, batch_size):
        """使用PIL逐张resize，带进度条"""
        pbar = comfy.utils.ProgressBar(batch_size)
        results = []
        for i in range(batch_size):
            img_tensor = image[i]
            pil_img = Image.fromarray((img_tensor.cpu().numpy() * 255).astype(np.uint8))
            resized_pil = pil_img.resize((target_w, target_h), pil_filter)
            resized_tensor = torch.from_numpy(np.array(resized_pil).astype(np.float32) / 255.0)
            results.append(resized_tensor)
            pbar.update_absolute(i + 1, batch_size)
        return torch.stack(results)

    def _resize_rtx_batch(self, image, target_w, target_h, batch_size):
        """RTX+lanczos批量处理，自适应分批，带进度条"""
        try:
            import nvvfx
        except ImportError:
            raise ImportError("nvvfx库未安装，无法使用RTX缩放。请安装NVIDIA RTX VSR。")

        # 自适应分批：参照Nvidia RTX Nodes，限制每批输出不超过16MP
        MAX_PIXELS = 1024 * 1024 * 16
        out_pixels = target_w * target_h
        chunk_size = max(1, MAX_PIXELS // max(1, out_pixels))

        pbar = comfy.utils.ProgressBar(batch_size)

        # 计算RTX内部尺寸
        orig_h, orig_w = image.shape[1], image.shape[2]
        if target_w >= orig_w and target_h >= orig_h:
            rtx_w, rtx_h = target_w, target_h
        else:
            rtx_w = max(target_w, orig_w * 2)
            rtx_h = max(target_h, orig_h * 2)
        rtx_w = max(8, round(rtx_w / 8) * 8)
        rtx_h = max(8, round(rtx_h / 8) * 8)

        quality_mapping = {
            "ULTRA": nvvfx.effects.QualityLevel.ULTRA,
            "HIGH": nvvfx.effects.QualityLevel.HIGH,
            "MEDIUM": nvvfx.effects.QualityLevel.MEDIUM,
            "LOW": nvvfx.effects.QualityLevel.LOW,
        }

        upscaled_chunks = []
        processed = 0

        # 单次创建RTX context，复用处理所有帧
        with nvvfx.VideoSuperRes(quality_mapping["ULTRA"]) as sr:
            sr.output_width = rtx_w
            sr.output_height = rtx_h
            sr.load()

            for i in range(0, batch_size, chunk_size):
                batch = image[i:i + chunk_size]
                # BHWC -> BCHW, 移到GPU
                batch_cuda = batch.permute(0, 3, 1, 2).cuda().contiguous()

                chunk_outputs = []
                for j in range(batch_cuda.shape[0]):
                    dlpack_out = sr.run(batch_cuda[j]).image
                    rtx_output = torch.from_dlpack(dlpack_out).clone()
                    # CHW -> HWC
                    chunk_outputs.append(rtx_output.permute(1, 2, 0).cpu())
                    processed += 1
                    pbar.update_absolute(processed, batch_size)

                chunk_tensor = torch.stack(chunk_outputs, dim=0)

                # 如果RTX输出尺寸与目标不同，用lanczos缩小
                if rtx_w != target_w or rtx_h != target_h:
                    # 用GPU批量lanczos缩小
                    chunk_tensor = self._resize_torch_batch(chunk_tensor, target_w, target_h, "bicubic")

                upscaled_chunks.append(chunk_tensor)

        return torch.cat(upscaled_chunks, dim=0)


NODE_CLASS_MAPPINGS.update({
    "PixelCountScaler": PixelCountScaler,
})

NODE_DISPLAY_NAME_MAPPINGS.update({
    "PixelCountScaler": "🖼️Pixel Count Scaler"
})


# Keyword Filter Node - 从输入文本中提取匹配的关键词
class KeywordFilter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True}),
                "keywords": ("STRING", {"default": "", "multiline": True, "placeholder": "每行一个关键词"}),
                "case_sensitive": ("BOOLEAN", {"default": False}),
                "separator": ("STRING", {"default": ", "}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("matched_keywords", "matched_text")
    FUNCTION = "filter_keywords"
    CATEGORY = "text"

    def filter_keywords(self, text: str, keywords: str, case_sensitive: bool, separator: str) -> tuple:
        # 解析关键词列表，每行一个
        keyword_list = [kw.strip() for kw in keywords.split('\n') if kw.strip()]

        if not keyword_list:
            return ("", "")

        search_text = text if case_sensitive else text.lower()
        matched = []
        matched_original = []

        for kw in keyword_list:
            kw_search = kw if case_sensitive else kw.lower()
            if kw_search in search_text:
                if kw not in matched:
                    matched.append(kw)
                    # 从原始文本中提取匹配的片段
                    if case_sensitive:
                        idx = text.find(kw)
                    else:
                        idx = text.lower().find(kw.lower())
                    if idx != -1:
                        matched_original.append(text[idx:idx + len(kw)])

        matched_keywords = separator.join(matched)
        matched_text = separator.join(matched_original)

        return (matched_keywords, matched_text)


NODE_CLASS_MAPPINGS.update({
    "KeywordFilter": KeywordFilter,
})

NODE_DISPLAY_NAME_MAPPINGS.update({
    "KeywordFilter": "🏷️ Keyword Filter"
})


# Keyword Filter + LoRA 条件加载节点
class KeywordFilterLoRA:
    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True}),
                "keywords": ("STRING", {"default": "", "multiline": True, "placeholder": "每行一个关键词"}),
                "case_sensitive": ("BOOLEAN", {"default": False}),
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("STRING", "MODEL", "CLIP", "STRING")
    RETURN_NAMES = ("matched_keywords", "MODEL", "CLIP", "status")
    FUNCTION = "filter_and_load_lora"
    CATEGORY = "text"

    def filter_and_load_lora(self, text, keywords, case_sensitive, model, clip, lora_name, strength_model, strength_clip):
        # 解析关键词
        keyword_list = [kw.strip() for kw in keywords.split('\n') if kw.strip()]

        # 匹配关键词
        search_text = text if case_sensitive else text.lower()
        matched = []
        for kw in keyword_list:
            kw_search = kw if case_sensitive else kw.lower()
            if kw_search in search_text and kw not in matched:
                matched.append(kw)

        matched_keywords = ", ".join(matched)

        # 无匹配，透传原始 MODEL/CLIP
        if not matched:
            return (matched_keywords, model, clip, "No match, LoRA not loaded")

        # 有关键词匹配，加载 LoRA
        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        lora = None
        if self.loaded_lora is not None:
            if self.loaded_lora[0] == lora_path:
                lora = self.loaded_lora[1]
            else:
                self.loaded_lora = None

        if lora is None:
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            self.loaded_lora = (lora_path, lora)

        model_lora, clip_lora = comfy.sd.load_lora_for_models(model, clip, lora, strength_model, strength_clip)
        status = f"LoRA loaded: {lora_name} (model={strength_model}, clip={strength_clip})"
        return (matched_keywords, model_lora, clip_lora, status)


class OutpaintingPreprocess:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "expand_top": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "expand_bottom": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "expand_left": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "expand_right": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "fill_color": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFF, "step": 1}),
                "auto_color": ("BOOLEAN", {"default": False}),
                "auto_color_sample": ("INT", {"default": 5, "min": 1, "max": 64, "step": 1}),
                "feather_radius": ("INT", {"default": 0, "min": 0, "max": 256, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("image", "mask", "expand_params")
    FUNCTION = "expand_image"
    CATEGORY = "image"

    def expand_image(self, image, expand_top, expand_bottom, expand_left, expand_right, fill_color, auto_color, auto_color_sample, feather_radius):
        batch = image.shape[0]
        h, w = image.shape[1], image.shape[2]

        # 计算各方向扩展像素
        top_px = int(h * expand_top)
        bottom_px = int(h * expand_bottom)
        left_px = int(w * expand_left)
        right_px = int(w * expand_right)

        new_h = h + top_px + bottom_px
        new_w = w + left_px + right_px

        # 原图在画布中的位置
        ox, oy = left_px, top_px

        results_img = []
        results_mask = []

        for bi in range(batch):
            img_np = (image[bi].cpu().numpy() * 255).astype(np.uint8)

            # 创建原图图层: 用 BORDER_REPLICATE 将边缘像素延伸到扩展区域，避免黑色
            canvas_orig = cv2.copyMakeBorder(
                img_np, top_px, bottom_px, left_px, right_px, cv2.BORDER_REPLICATE
            ).astype(np.float32)

            # 创建填充画布，并向原图区域延伸填充色，避免 alpha 过渡时出现黑色
            if auto_color:
                sample = auto_color_sample
                top_color = img_np[:sample, :, :].mean(axis=(0, 1)) if top_px > 0 else None
                bottom_color = img_np[-sample:, :, :].mean(axis=(0, 1)) if bottom_px > 0 else None
                left_color = img_np[:, :sample, :].mean(axis=(0, 1)) if left_px > 0 else None
                right_color = img_np[:, -sample:, :].mean(axis=(0, 1)) if right_px > 0 else None

                canvas_fill = np.zeros((new_h, new_w, 3), dtype=np.float32)

                # 填充各方向扩展区域
                if top_px > 0:
                    canvas_fill[:oy, :] = top_color
                if bottom_px > 0:
                    canvas_fill[oy + h:, :] = bottom_color
                if left_px > 0:
                    canvas_fill[:, :ox] = left_color
                if right_px > 0:
                    canvas_fill[:, ox + w:] = right_color

                # 填充四角: 两边颜色的平均值
                if top_px > 0 and left_px > 0:
                    canvas_fill[:oy, :ox] = (top_color + left_color) / 2.0
                if top_px > 0 and right_px > 0:
                    canvas_fill[:oy, ox + w:] = (top_color + right_color) / 2.0
                if bottom_px > 0 and left_px > 0:
                    canvas_fill[oy + h:, :ox] = (bottom_color + left_color) / 2.0
                if bottom_px > 0 and right_px > 0:
                    canvas_fill[oy + h:, ox + w:] = (bottom_color + right_color) / 2.0

                # 向原图区域延伸各方向的填充色
                if top_px > 0:
                    canvas_fill[oy:oy + h, ox:ox + w] = top_color
                if bottom_px > 0:
                    canvas_fill[oy:oy + h, ox:ox + w] = bottom_color
                if left_px > 0:
                    canvas_fill[oy:oy + h, ox:ox + w] = left_color
                if right_px > 0:
                    canvas_fill[oy:oy + h, ox:ox + w] = right_color
                # 多方向时原图区域取所有方向平均
                active_colors = [c for c in [top_color, bottom_color, left_color, right_color] if c is not None]
                if len(active_colors) > 1:
                    canvas_fill[oy:oy + h, ox:ox + w] = np.mean(active_colors, axis=0)

                # 四角到直边的颜色过渡羽化
                if feather_radius > 0:
                    canvas_blur = cv2.GaussianBlur(canvas_fill, (feather_radius * 2 + 1, feather_radius * 2 + 1), 0)
                    corner_mask = np.zeros((new_h, new_w), dtype=bool)
                    if top_px > 0:
                        corner_mask[:oy, :] = True
                    if bottom_px > 0:
                        corner_mask[oy + h:, :] = True
                    feather_band = feather_radius
                    if left_px > 0:
                        bl = max(0, ox - feather_band)
                        br = min(ox + feather_band, new_w)
                        if top_px > 0:
                            corner_mask[:oy, bl:br] = True
                        if bottom_px > 0:
                            corner_mask[oy + h:, bl:br] = True
                    if right_px > 0:
                        bl = max(0, ox + w - feather_band)
                        br = min(ox + w + feather_band, new_w)
                        if top_px > 0:
                            corner_mask[:oy, bl:br] = True
                        if bottom_px > 0:
                            corner_mask[oy + h:, bl:br] = True
                    if top_px > 0:
                        bt = max(0, oy - feather_band)
                        bb = min(oy + feather_band, new_h)
                        if left_px > 0:
                            corner_mask[bt:bb, :ox] = True
                        if right_px > 0:
                            corner_mask[bt:bb, ox + w:] = True
                    if bottom_px > 0:
                        bt = max(0, oy + h - feather_band)
                        bb = min(oy + h + feather_band, new_h)
                        if left_px > 0:
                            corner_mask[bt:bb, :ox] = True
                        if right_px > 0:
                            corner_mask[bt:bb, ox + w:] = True
                    canvas_fill[corner_mask] = canvas_blur[corner_mask]
            else:
                r = (fill_color >> 16) & 0xFF
                g = (fill_color >> 8) & 0xFF
                b = fill_color & 0xFF
                canvas_fill = np.full((new_h, new_w, 3), [r, g, b], dtype=np.float32)

            # 创建混合 alpha: 1.0=原图, 0.0=填充
            alpha = np.zeros((new_h, new_w), dtype=np.float32)
            alpha[oy:oy + h, ox:ox + w] = 1.0

            # 羽化 alpha: 边界处互相渐变
            if feather_radius > 0:
                alpha = cv2.GaussianBlur(alpha, (feather_radius * 2 + 1, feather_radius * 2 + 1), 0)

            # 混合: 原图 * alpha + 填充 * (1 - alpha)
            alpha_3d = alpha[:, :, np.newaxis]
            canvas = canvas_orig * alpha_3d + canvas_fill * (1.0 - alpha_3d)
            canvas = np.clip(canvas, 0, 255).astype(np.uint8)

            # 创建 mask 输出: 1.0=扩展区域, 0.0=原图区域
            mask = np.zeros((new_h, new_w), dtype=np.float32)
            if top_px > 0:
                mask[:oy, :] = 1.0
            if bottom_px > 0:
                mask[oy + h:, :] = 1.0
            if left_px > 0:
                mask[:, :ox] = 1.0
            if right_px > 0:
                mask[:, ox + w:] = 1.0

            # mask 羽化: 仅内接边缘羽化，外框边缘不羽化
            if feather_radius > 0:
                blurred = cv2.GaussianBlur(mask, (feather_radius * 2 + 1, feather_radius * 2 + 1), 0)
                border_fix = np.zeros_like(mask, dtype=bool)
                if top_px > 0:
                    border_fix[:min(feather_radius, top_px), :] = True
                if bottom_px > 0:
                    border_fix[max(0, new_h - feather_radius):, :] = True
                if left_px > 0:
                    border_fix[:, :min(feather_radius, left_px)] = True
                if right_px > 0:
                    border_fix[:, max(0, new_w - feather_radius):] = True
                outer_edge = border_fix & (mask == 1.0)
                blurred[outer_edge] = 1.0
                mask = blurred

            results_img.append(torch.from_numpy(canvas.astype(np.float32) / 255.0).unsqueeze(0))
            results_mask.append(torch.from_numpy(mask).unsqueeze(0))

        out_img = torch.cat(results_img, dim=0)
        out_mask = torch.cat(results_mask, dim=0)

        expand_params = f"{w},{h},{expand_top},{expand_bottom},{expand_left},{expand_right}"

        return (out_img, out_mask, expand_params)


class OutpaintingRemove:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "expand_params": ("STRING", {"default": "", "forceInput": True}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "remove_outpainting"
    CATEGORY = "image"

    def remove_outpainting(self, image, expand_params):
        # 解析参数: 原始w,原始h,expand_top,expand_bottom,expand_left,expand_right
        parts = expand_params.split(",")
        orig_w, orig_h = int(parts[0]), int(parts[1])
        e_top, e_bottom = float(parts[2]), float(parts[3])
        e_left, e_right = float(parts[4]), float(parts[5])

        batch = image.shape[0]
        cur_h, cur_w = image.shape[1], image.shape[2]

        # 原始扩图后的总尺寸
        expanded_w = orig_w * (1 + e_left + e_right)
        expanded_h = orig_h * (1 + e_top + e_bottom)

        # 按比例计算原图在当前图像中的位置
        left_px = int(e_left / (1 + e_left + e_right) * cur_w)
        top_px = int(e_top / (1 + e_top + e_bottom) * cur_h)
        right_px = cur_w - int(e_right / (1 + e_left + e_right) * cur_w)
        bottom_px = cur_h - int(e_bottom / (1 + e_top + e_bottom) * cur_h)

        # 确保边界合法
        left_px = max(0, left_px)
        top_px = max(0, top_px)
        right_px = min(cur_w, right_px)
        bottom_px = min(cur_h, bottom_px)

        results = []
        for i in range(batch):
            cropped = image[i, top_px:bottom_px, left_px:right_px, :]
            results.append(cropped.unsqueeze(0))

        return (torch.cat(results, dim=0),)


class KeywordReverseFilter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True}),
                "keywords": ("STRING", {"default": "", "multiline": True, "placeholder": "每行一个关键词"}),
                "output_text": ("STRING", {"default": "", "multiline": True, "placeholder": "不匹配时输出的文本"}),
                "case_sensitive": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("result", "status")
    FUNCTION = "reverse_filter"
    CATEGORY = "text"

    def reverse_filter(self, text, keywords, output_text, case_sensitive):
        keyword_list = [kw.strip() for kw in keywords.split('\n') if kw.strip()]

        if not keyword_list:
            return ("", "No keywords specified")

        search_text = text if case_sensitive else text.lower()

        for kw in keyword_list:
            kw_search = kw if case_sensitive else kw.lower()
            if kw_search in search_text:
                return ("", f"Matched: {kw}")

        return (output_text, "No match, output text sent")


class DarkRegionOverlay:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
                "brightness_threshold": ("FLOAT", {"default": 0.10, "min": 0.0, "max": 1.0, "step": 0.01}),
                "overlay_color": ("INT", {"default": 0xFFFFFF, "min": 0, "max": 0xFFFFFF, "step": 1}),
                "feather_radius": ("INT", {"default": 5, "min": 0, "max": 256, "step": 1}),
                "dilate_pixels": ("INT", {"default": 0, "min": 0, "max": 512, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "overlay_dark_regions"
    CATEGORY = "image"

    def overlay_dark_regions(self, image_a, image_b, brightness_threshold, overlay_color, feather_radius, dilate_pixels):
        batch_a = image_a.shape[0]
        h_a, w_a = image_a.shape[1], image_a.shape[2]

        # 解析覆盖色
        r = (overlay_color >> 16) & 0xFF
        g = (overlay_color >> 8) & 0xFF
        b = overlay_color & 0xFF

        results_img = []
        results_mask = []

        for i in range(batch_a):
            img_a = (image_a[i].cpu().numpy() * 255).astype(np.float32)

            # 取 B 图对应帧，若 batch 不够则循环
            idx_b = i % image_b.shape[0]
            img_b = (image_b[idx_b].cpu().numpy() * 255).astype(np.float32)

            # 将 B 图 resize 到 A 图尺寸
            if img_b.shape[0] != h_a or img_b.shape[1] != w_a:
                img_b_pil = Image.fromarray(img_b.astype(np.uint8))
                img_b_pil = img_b_pil.resize((w_a, h_a), Image.LANCZOS)
                img_b = np.array(img_b_pil).astype(np.float32)

            # 计算 B 图亮度
            brightness = img_b.mean(axis=2) / 255.0

            # 暗区蒙版: 亮度低于阈值的区域为 1.0
            mask = np.zeros((h_a, w_a), dtype=np.float32)
            mask[brightness < brightness_threshold] = 1.0

            # 蒙版扩张
            if dilate_pixels > 0:
                kernel = np.ones((dilate_pixels * 2 + 1, dilate_pixels * 2 + 1), dtype=np.uint8)
                mask_uint8 = (mask * 255).astype(np.uint8)
                mask_uint8 = cv2.dilate(mask_uint8, kernel, iterations=1)
                mask = mask_uint8.astype(np.float32) / 255.0

            # 羽化蒙版
            if feather_radius > 0:
                mask = cv2.GaussianBlur(mask, (feather_radius * 2 + 1, feather_radius * 2 + 1), 0)

            # 用蒙版覆盖 A 图: result = A * (1-mask) + color * mask
            mask_3d = mask[:, :, np.newaxis]
            overlay = np.full_like(img_a, [r, g, b], dtype=np.float32)
            result = img_a * (1.0 - mask_3d) + overlay * mask_3d
            result = np.clip(result, 0, 255).astype(np.uint8)

            results_img.append(torch.from_numpy(result.astype(np.float32) / 255.0).unsqueeze(0))
            results_mask.append(torch.from_numpy(mask).unsqueeze(0))

        out_img = torch.cat(results_img, dim=0)
        out_mask = torch.cat(results_mask, dim=0)

        return (out_img, out_mask)


class DanbooruFetcher:
    """Fetch tags and image from a Danbooru post URL (supports images, videos, gifs, etc.)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING", {"default": "", "multiline": False}),
            },
            "optional": {
                "tag_separator": ("STRING", {"default": ", ", "multiline": False}),
                "replace_underscores": ("BOOLEAN", {"default": True}),
                "exclude_general": ("BOOLEAN", {"default": False}),
                "exclude_meta": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("image", "all_tags", "copyright_tags", "character_tags", "artist_tags", "general_tags", "meta_tags")
    FUNCTION = "fetch"
    CATEGORY = "Text"
    OUTPUT_NODE = True

    @staticmethod
    def _fetch_json(api_url):
        """Fetch JSON from Danbooru API. Tries curl first (bypasses Cloudflare), then requests."""
        import subprocess
        import json

        # Method 1: curl (bypasses Cloudflare bot detection)
        try:
            result = subprocess.run(
                ["curl", "-s", "-H", "Accept: application/json", api_url],
                capture_output=True, text=True, timeout=15,
            )
            if result.returncode == 0 and result.stdout.strip().startswith("{"):
                return json.loads(result.stdout)
        except Exception:
            pass

        # Method 2: requests
        import requests
        try:
            resp = requests.get(api_url, headers={"Accept": "application/json"}, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except Exception:
            pass

        # Method 3: cloudscraper
        try:
            import cloudscraper
            scraper = cloudscraper.create_scraper()
            resp = scraper.get(api_url, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except Exception:
            pass

        return None

    @staticmethod
    def _download_file(url, timeout=30):
        """Download a file from URL, return bytes. Tries curl then requests."""
        import subprocess
        # curl
        try:
            result = subprocess.run(
                ["curl", "-s", url],
                capture_output=True, timeout=timeout,
            )
            if result.returncode == 0 and len(result.stdout) > 100:
                return result.stdout
        except Exception:
            pass
        # requests
        import requests
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            return resp.content
        except Exception:
            pass
        return None

    @staticmethod
    def _image_to_tensor(img):
        """Convert a PIL Image (RGB) to ComfyUI IMAGE tensor."""
        return torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0)

    def _fetch_image(self, data):
        """Download the post image. For videos, extract first frame via cv2."""
        import tempfile
        import io

        file_ext = data.get("file_ext", "").lower()
        file_url = data.get("file_url", "")

        if not file_url:
            return None

        raw = self._download_file(file_url)
        if not raw:
            return None

        image_exts = {"jpg", "jpeg", "png", "webp"}
        video_exts = {"mp4", "webm", "avi", "mkv", "gif"}

        if file_ext in image_exts:
            img = Image.open(io.BytesIO(raw)).convert("RGB")
            return self._image_to_tensor(img)

        if file_ext in video_exts:
            tmp = tempfile.NamedTemporaryFile(suffix=f".{file_ext}", delete=False)
            try:
                tmp.write(raw)
                tmp.close()
                if file_ext == "gif":
                    img = Image.open(tmp.name).convert("RGB")
                    return self._image_to_tensor(img)
                else:
                    cap = cv2.VideoCapture(tmp.name)
                    ret, frame = cap.read()
                    cap.release()
                    if ret:
                        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        return self._image_to_tensor(img)
            finally:
                os.unlink(tmp.name)

        return None

    def fetch(self, url, tag_separator=", ", replace_underscores=True, exclude_general=False, exclude_meta=False):
        import re

        empty_strings = ("", "", "", "", "", "")
        empty_black = (torch.zeros(1, 64, 64, 3, dtype=torch.float32),) + empty_strings

        if not url or "danbooru.donmai.us" not in url:
            return empty_black

        match = re.search(r'/posts/(\d+)', url)
        if not match:
            return empty_black

        post_id = match.group(1)
        api_url = f"https://danbooru.donmai.us/posts/{post_id}.json"

        data = self._fetch_json(api_url)
        if not data:
            return empty_black

        # Image
        image_tensor = self._fetch_image(data)
        if image_tensor is None:
            image_tensor = torch.zeros(1, 64, 64, 3, dtype=torch.float32)

        # Tags
        def clean_tag(s):
            tags = s.split()
            if replace_underscores:
                tags = [t.replace("_", " ") for t in tags]
            return tag_separator.join(tags)

        tag_copyright = clean_tag(data.get("tag_string_copyright", ""))
        tag_character = clean_tag(data.get("tag_string_character", ""))
        tag_artist = clean_tag(data.get("tag_string_artist", ""))
        tag_general = clean_tag(data.get("tag_string_general", ""))
        tag_meta = clean_tag(data.get("tag_string_meta", ""))

        parts = []
        if tag_copyright:
            parts.append(tag_copyright)
        if tag_character:
            parts.append(tag_character)
        if tag_artist:
            parts.append(tag_artist)
        if tag_general and not exclude_general:
            parts.append(tag_general)
        if tag_meta and not exclude_meta:
            parts.append(tag_meta)
        all_tags = tag_separator.join(parts)

        return (image_tensor, all_tags, tag_copyright, tag_character, tag_artist, tag_general, tag_meta)


NODE_CLASS_MAPPINGS.update({
    "DanbooruFetcher": DanbooruFetcher,
})

NODE_DISPLAY_NAME_MAPPINGS.update({
    "DanbooruFetcher": "🏷️ Danbooru Fetcher",
})


NODE_CLASS_MAPPINGS.update({
    "KeywordFilterLoRA": KeywordFilterLoRA,
    "OutpaintingPreprocess": OutpaintingPreprocess,
    "OutpaintingRemove": OutpaintingRemove,
    "KeywordReverseFilter": KeywordReverseFilter,
    "DarkRegionOverlay": DarkRegionOverlay,
})

NODE_DISPLAY_NAME_MAPPINGS.update({
    "KeywordFilterLoRA": "🏷️ Keyword Filter + LoRA",
    "OutpaintingPreprocess": "🖼️ Outpainting Preprocess",
    "OutpaintingRemove": "🖼️ Outpainting Remove",
    "KeywordReverseFilter": "🏷️ Keyword Reverse Filter",
    "DarkRegionOverlay": "🖼️ Dark Region Overlay",
})


# Text File Line Reader
class TextFileLineReader:
    """Read a txt file and output a specific line by index. Index updates after each run based on mode."""

    _index_state = {}  # abs_path -> current_index (persists across runs)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {"default": "", "multiline": False}),
                "index": ("INT", {"default": 0, "min": 0, "max": 1000000}),
                "mode": (["fix", "increment", "decrement", "random"],),
                "reset": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("text", "current_index")
    FUNCTION = "read_line"
    CATEGORY = "Text"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, file_path, index, mode, reset):
        # Force re-execution every time so increment/decrement/random state updates
        return time.time()

    def read_line(self, file_path, index, mode, reset):
        import random

        if not file_path or not os.path.isfile(file_path):
            return ("", index)

        with open(file_path, "r", encoding="utf-8") as f:
            lines = [line.rstrip("\n") for line in f.readlines()]

        if not lines:
            return ("", index)

        total = len(lines)
        abs_path = os.path.abspath(file_path)

        # Determine current index
        if reset or mode == "fix":
            current_index = index
        else:
            current_index = self._index_state.get(abs_path, index)

        # Clamp to valid range
        current_index = max(0, min(current_index, total - 1))
        text = lines[current_index]

        # Update state for next run
        if mode == "increment":
            self._index_state[abs_path] = (current_index + 1) % total
        elif mode == "decrement":
            self._index_state[abs_path] = (current_index - 1) % total
        elif mode == "random":
            self._index_state[abs_path] = random.randint(0, total - 1)

        return (text, current_index)


NODE_CLASS_MAPPINGS.update({
    "TextFileLineReader": TextFileLineReader,
})

NODE_DISPLAY_NAME_MAPPINGS.update({
    "TextFileLineReader": "📄 Text File Line Reader",
})


# ============================================================================
# Prompt Enhance (OpenAI-compatible) —— 设置节点 + 提示词增强节点
# 配置记录保存在插件目录下的 prompt_enhance_endpoints.json
# ============================================================================

# 配置记录文件路径（与 ray_nodes.py 同目录）
PROMPT_ENHANCE_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompt_enhance_endpoints.json")


def _load_prompt_enhance_records():
    """读取 JSON 配置记录；文件不存在或损坏时返回空 dict。"""
    if not os.path.isfile(PROMPT_ENHANCE_JSON):
        return {}
    try:
        with open(PROMPT_ENHANCE_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {}


def _save_prompt_enhance_records(records):
    """写入 JSON 配置记录（UTF-8、缩进、保留中文可读）。"""
    try:
        with open(PROMPT_ENHANCE_JSON, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _join_openai_path(base_url, path):
    """拼接 OpenAI 兼容路径。

    若 base_url 已带版本段（/v1、/v2、/v3 等）则直接拼接，
    否则补上 /v1（兼容 OpenAI 及大部分第三方网关，如豆包 /api/coding/v3 已带版本段则保留）。
    """
    url = (base_url or "").rstrip("/")
    if re.search(r"/v\d+$", url):
        return f"{url}/{path}"
    return f"{url}/v1/{path}"


def _extract_content(data):
    """从 OpenAI 兼容响应里取文本内容。

    兼容思考型模型（如 Gemini 2.5/3 Pro、DeepSeek-R 系）：当 message.content 为空时，
    回退读取 reasoning_content，否则会拿到空字符串导致节点输出空白。
    """
    try:
        msg = data["choices"][0]["message"]
    except Exception:
        return ""
    content = (msg.get("content") or "").strip()
    if content:
        return content
    # 回退：思考型模型可能把所有输出放在 reasoning_content 里
    return (msg.get("reasoning_content") or "").strip()


def _test_endpoint_connectivity(base_url, api_key, model_id="", timeout=15):
    """测试 OpenAI 兼容端点连通性：GET {base_url}/models。

    返回 (ok: bool, message: str)。
    """
    import requests

    # 若 base_url 已带版本段（/v1、/v2、/v3 ...）则直接拼接，
    # 否则补上 /v1（兼容 OpenAI 及大部分第三方网关）。
    models_url = _join_openai_path(base_url, "models")

    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        resp = requests.get(models_url, headers=headers, timeout=timeout)
        if resp.status_code == 200:
            return True, "Connected (200 OK)"
        # 401/403 一般代表 key 问题，但端点本身可达
        return False, f"HTTP {resp.status_code}: {resp.text[:120]}"
    except requests.exceptions.Timeout:
        return False, "Timeout"
    except requests.exceptions.ConnectionError as e:
        return False, f"Connection error: {e}"
    except Exception as e:
        return False, f"Error: {e}"


class PromptEnhanceSettings:
    """提示词扩写 —— 端点设置节点。

    输入名称、URL、API Key、默认模型；运行时测试连通性并写入 JSON 记录。
    force_save 开关允许测试失败也追加/更新记录。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "name": ("STRING", {"default": "", "placeholder": "记录名称（同名即更新）"}),
                "base_url": ("STRING", {"default": "https://api.openai.com/v1", "placeholder": "OpenAI 兼容端点，如 https://api.openai.com/v1"}),
                "api_key": ("STRING", {"default": "", "placeholder": "sk-..."}),
                "model_id": ("STRING", {"default": "", "placeholder": "默认模型 id，如 gpt-4o-mini"}),
                "force_save": ("BOOLEAN", {"default": False, "label_on": "测试失败也保存", "label_off": "仅测试成功保存"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "apply"
    CATEGORY = "text"
    OUTPUT_NODE = True

    def apply(self, name, base_url, api_key, model_id, force_save):
        name = (name or "").strip()
        base_url = (base_url or "").strip()
        api_key = api_key or ""
        model_id = (model_id or "").strip()

        if not name:
            return ("❌ 记录名称不能为空",)

        records = _load_prompt_enhance_records()
        ok, msg = _test_endpoint_connectivity(base_url, api_key, model_id)

        now = time.strftime("%Y-%m-%d %H:%M:%S")
        records[name] = {
            "url": base_url,
            "api_key": api_key,
            "model_id": model_id,
            "status": f"{'OK' if ok else 'FAIL'}: {msg}",
            "last_tested": now,
        }

        if ok or force_save:
            _save_prompt_enhance_records(records)
            if ok:
                status = f"✅ 连通成功，已保存记录「{name}」({msg})"
            else:
                status = f"⚠️ 连通失败，但已保存记录「{name}」（force_save）\n原因：{msg}"
        else:
            status = f"❌ 连通失败，未保存记录「{name}」\n原因：{msg}\n（如需强制保存，请打开 force_save 开关）"

        return (status,)


class PromptEnhancer:
    """提示词扩写 —— 增强节点。

    从 JSON 记录中选择一个端点，输入模型 id、系统提示词、用户提示词，
    以 OpenAI 标准 chat/completions 格式请求并返回扩写结果。
    支持内置预设模板（画面要素位置 / 细节要素 / 风格 / 光照），
    并通过 seed 实现确定性：输入与 seed 未变时复用缓存，避免重复请求 LLM。
    """

    # 内置预设提示词模板。每个模板的 system_prompt 覆盖
    # 「画面要素位置 / 细节要素 / 风格 / 光照」四个维度。
    # 选 "(不使用预设)" 时使用用户自定义的 system_prompt。
    PRESETS = {
        "(不使用预设)": {
            "system_prompt": "",
            "description": "完全使用下方自定义系统提示词",
        },
        "🎨 通用扩写（推荐）": {
            "system_prompt": (
                "你是一位专业的 AI 绘画提示词工程师。请把用户给出的原始提示词扩写为一段高质量的英文画面描述，"
                "严格按以下四个维度组织，且每个维度都不可遗漏：\n"
                "\n"
                "1. 画面要素位置 (composition & layout)：主体在画面中的位置、构图方式（如居中/三分构图/仰视）、"
                "镜头（如特写/全身/广角）、画面景别、前景/背景层次。\n"
                "2. 细节要素 (details)：人物/物体的具体特征——五官、表情、发色发型、服饰、配饰、材质纹理、"
                "动作姿态，以及环境中的道具与背景物体。\n"
                "3. 风格 (style)：画风/艺术流派（如写实/二次元/油画/赛博朋克）、画家或作品参考、"
                "色彩倾向（冷暖/高饱和/低饱和）、渲染质感。\n"
                "4. 光照 (lighting)：光源类型（自然光/逆光/侧光/体积光）、光质（硬光/柔光）、"
                "阴影方向与对比、氛围色调。\n"
                "\n"
                "要求：\n"
                "- 只输出最终的英文提示词（danbooru 风格标签或自然语言均可，保持与原文一致的风格），不要解释。\n"
                "- 保留用户原文的核心语义与关键标签，只做增补，不删改原意。\n"
                "- 输出为逗号分隔的英文短语，长度适中，避免冗长重复。"
            ),
            "description": "覆盖四个维度，输出逗号分隔英文标签",
        },
        "🧸 二次元角色 (Anime Character)": {
            "system_prompt": (
                "你是二次元插画提示词扩写专家。把用户的原始提示词扩写为英文标签序列，覆盖以下四维度：\n"
                "1. 画面要素位置：人物在画面的位置与构图（如 1girl, upper body, cowboy shot, from above）、"
                "镜头景别。\n"
                "2. 细节要素：角色五官/瞳色/发色发型/表情/服装/配饰/姿态/动作，使用 danbooru 标签。\n"
                "3. 风格：画风（如 anime, illustration, masterpiece, best quality）、参考画师、色彩倾向。\n"
                "4. 光照：光源与氛围（如 soft lighting, backlighting, rim light, dramatic lighting）。\n"
                "要求：输出逗号分隔英文 danbooru 标签；保留原文核心标签；不解释、不输出多余文字。"
            ),
            "description": "二次元角色向，danbooru 标签风格",
        },
        "📷 写实摄影 (Photorealistic)": {
            "system_prompt": (
                "你是写实摄影提示词扩写专家。把用户的原始提示词扩写为一段英文自然语言描述，覆盖：\n"
                "1. 画面要素位置：主体位置、构图（rule of thirds, centered, leading lines）、焦段（如 85mm, wide angle）、"
                "景深与背景。\n"
                "2. 细节要素：主体的具体细节、皮肤质感、服饰材质、环境道具。\n"
                "3. 风格：照片类型与相机参考（如 DSLR, 35mm film, Fujifilm, bokeh, sharp focus, 8k, ultra detailed）。\n"
                "4. 光照：光线（golden hour, soft window light, studio lighting, volumetric light）、阴影与对比度。\n"
                "要求：输出一段连贯的英文描述，保留原文核心语义，不解释。"
            ),
            "description": "写实摄影向，自然语言描述",
        },
        "🌅 场景概念 (Scene Concept)": {
            "system_prompt": (
                "你是场景概念设计提示词扩写专家。把用户的原始场景提示词扩写为英文描述，覆盖四维度：\n"
                "1. 画面要素位置：场景构图、视角（eye-level, bird's eye view）、前景/中景/远景层次、画面纵深。\n"
                "2. 细节要素：建筑/自然物体的结构、材质、植被、天气、点缀元素与人物（如有）。\n"
                "3. 风格：概念艺术风格（concept art, matte painting, epic, detailed）、参考画师、色彩方案。\n"
                "4. 光照：时间与天气光线（dawn, dusk, overcast, dramatic sky, god rays）、氛围。\n"
                "要求：输出英文描述，保留原文核心语义，不解释。"
            ),
            "description": "场景概念设计向",
        },
    }

    @classmethod
    def get_preset_names(cls):
        return list(cls.PRESETS.keys())

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "endpoint": (cls.get_endpoints(),),
                "preset": (cls.get_preset_names(), {"default": "🎨 通用扩写（推荐）"}),
                "model_id": ("STRING", {"default": "", "placeholder": "留空则使用记录中的默认模型"}),
                "system_prompt": ("STRING", {"default": "", "multiline": True, "placeholder": "系统提示词（选了预设则叠加在预设之后；选\"不使用预设\"则单独生效）"}),
                "user_prompt": ("STRING", {"default": "", "multiline": True, "placeholder": "需要扩写的原始提示词"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05}),
                "max_tokens": ("INT", {"default": 1024, "min": 1, "max": 32768, "step": 1}),
            },
        }

    @classmethod
    def get_endpoints(cls):
        """动态下拉选项：读取 JSON 记录名；无记录时给兜底项避免加载失败。"""
        records = _load_prompt_enhance_records()
        names = list(records.keys())
        if not names:
            return ["(无记录，请先用设置节点添加)"]
        return names

    @classmethod
    def IS_CHANGED(cls, endpoint, preset, model_id, system_prompt, user_prompt, seed, **kwargs):
        # 确定性缓存键：端点记录内容 + 预设 + 模型 + 系统提示词 + 用户提示词 + seed。
        # 这些都未变时返回同一值 → ComfyUI 复用上次结果，跳过对 LLM 的重复请求。
        records = _load_prompt_enhance_records()
        rec_blob = json.dumps(records.get(endpoint, {}), ensure_ascii=False, sort_keys=True)
        key = "|".join([
            rec_blob,
            str(preset),
            str(model_id or ""),
            str(system_prompt or ""),
            str(user_prompt or ""),
            str(seed),
            str(kwargs.get("temperature", 0.7)),
            str(kwargs.get("max_tokens", 1024)),
        ])
        # 用稳定的 hash 作为缓存指纹（hexdigest 避免负数/平台差异）
        import hashlib
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("enhanced_text", "status")
    FUNCTION = "enhance"
    CATEGORY = "text"

    def enhance(self, endpoint, preset, model_id, system_prompt, user_prompt, seed, temperature=0.7, max_tokens=1024):
        import requests

        records = _load_prompt_enhance_records()
        if endpoint not in records:
            return ("", f"❌ 找不到记录「{endpoint}」，请先用设置节点添加")

        rec = records[endpoint]
        base_url = rec.get("url", "").strip()
        api_key = rec.get("api_key", "")
        use_model = (model_id or rec.get("model_id", "")).strip()

        if not use_model:
            return ("", "❌ 未指定模型 id（请填写 model_id 或在记录中设置默认模型）")

        if not base_url:
            return ("", f"❌ 记录「{endpoint}」缺少 url")

        # 合成最终系统提示词：预设模板在前，用户自定义在后（若都有则拼接）
        preset_sp = self.PRESETS.get(preset, {}).get("system_prompt", "")
        custom_sp = (system_prompt or "").strip()
        if preset_sp and custom_sp:
            effective_sp = preset_sp + "\n\n" + custom_sp
        else:
            effective_sp = preset_sp or custom_sp

        # 拼接 chat/completions 端点（已带 /vN 则保留，否则补 /v1）
        chat_url = _join_openai_path(base_url, "chat/completions")

        messages = []
        if effective_sp:
            messages.append({"role": "system", "content": effective_sp})
        # 把 seed 注入到用户消息前，使相同文本但不同 seed 也能触发新请求
        messages.append({"role": "user", "content": f"[seed={seed}]\n{user_prompt}"})

        payload = {
            "model": use_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        try:
            resp = requests.post(chat_url, headers=headers, json=payload, timeout=120)
        except requests.exceptions.Timeout:
            return ("", "❌ 请求超时")
        except Exception as e:
            return ("", f"❌ 请求异常：{e}")

        if resp.status_code != 200:
            return ("", f"❌ HTTP {resp.status_code}: {resp.text[:300]}")

        try:
            data = resp.json()
            content = _extract_content(data)
        except Exception as e:
            return ("", f"❌ 解析响应失败：{e}\n原始：{resp.text[:300]}")

        usage = data.get("usage", {})
        preset_tag = preset if preset != "(不使用预设)" else "自定义"
        status = (f"✅ 成功 | 端点: {endpoint} | 模型: {use_model} | 预设: {preset_tag} | "
                  f"seed: {seed} | tokens: {usage.get('total_tokens', '?')}")
        return (content, status)


def _image_to_data_url(image_tensor, quality=85, max_side=1024):
    """把单张 ComfyUI IMAGE 张量编码为 OpenAI Vision 用的 data URL。

    输入兼容 [H,W,C] 或 [1,H,W,C]（float 0-1），内部统一压成 [H,W,C]。
    - JPEG 压缩减少 token 消耗（Vision API 按图计费/计 token）
    - 长边缩放到 max_side，避免超大图超时
    """
    import base64
    from io import BytesIO

    arr = (255.0 * image_tensor.cpu().numpy()).clip(0, 255).astype(np.uint8)
    # 兜底：去掉可能的 batch 维度，确保 PIL 拿到的是 [H,W,C]
    if arr.ndim == 4:
        arr = arr[0]
    img = Image.fromarray(arr)
    # 长边等比缩小
    if max_side and max(img.size) > max_side:
        ratio = max_side / max(img.size)
        img = img.resize((int(img.size[0] * ratio), int(img.size[1] * ratio)), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _hash_image_tensor(image_tensor):
    """对单张图像张量算稳定指纹，用于缓存键。"""
    import hashlib
    arr = (255.0 * image_tensor.cpu().numpy()).clip(0, 255).astype(np.uint8)
    if arr.ndim == 4:
        arr = arr[0]
    return hashlib.sha256(arr.tobytes()).hexdigest()


class ImageAnalyzer:
    """图片分析节点（OpenAI Vision 兼容）。

    复用提示词扩写的 endpoint 记录，输入图像，输出按
    「画面要素 / 构图 / 光影 / 风格」描述的图片描述。
    内置预设系统提示词；seed 控制确定性（输入与 seed 未变则复用缓存）。
    """

    # 内置预设：按 画面要素 / 构图 / 光影 / 风格 描述图片
    PRESETS = {
        "(不使用预设)": {
            "system_prompt": "",
            "description": "完全使用下方自定义系统提示词",
        },
        "🔎 通用描述（推荐）": {
            "system_prompt": (
                "你是一位专业的图像描述专家。请仔细观察用户提供的图片，"
                "用英文输出一段结构化的图片描述，严格按以下四个维度组织，且每个维度都不可遗漏：\n"
                "\n"
                "1. 画面要素 (subject & elements)：画面中有哪些主体与物体，人物/动物/物品的数量、"
                "身份、表情、姿态、服饰、动作，以及环境里的道具与背景元素。\n"
                "2. 构图 (composition)：画面布局与取景，包括主体位置（居中/偏左/偏右）、"
                "构图方式（三分构图/对称/满版）、视角（俯视/仰视/平视）、镜头景别（特写/半身/全景/广角）、景深。\n"
                "3. 光影 (lighting)：光源类型与方向（自然光/逆光/侧光/顶光）、光质（硬光/柔光）、"
                "阴影、对比度、整体明暗与色调氛围。\n"
                "4. 风格 (style)：画面艺术风格（写实/二次元/油画/水彩/3D/赛博朋克等）、"
                "参考画师或作品、色彩倾向（冷暖/饱和度）、渲染与质感。\n"
                "\n"
                "要求：\n"
                "- 输出逗号分隔的英文短语（适合做 AI 绘画提示词），不要解释、不要编号。\n"
                "- 客观描述图中可见内容，不要编造图中没有的东西。\n"
                "- 简洁准确，避免冗长重复。\n"
                "- 【直白描述】如果画面含裸露、性、暴力、血腥等内容，必须用直白、准确、解剖学/医学术语直接描述"
                "（如 nude、penis、vagina、breasts、spreading、sexual intercourse、blood、wound），"
                "不得用「intimate」「revealing」「suggestive」「graphic」「mature content」这类含糊回避词代替。"
                "描述是为了让 AI 能复现画面，含糊会丢失语义。"
            ),
            "description": "四维度结构化描述，输出逗号分隔英文短语",
        },
        "🏷️ Danbooru 标签": {
            "system_prompt": (
                "你是 danbooru 风格的图像打标专家。观察图片，输出英文 danbooru 标签，覆盖：\n"
                "1. 画面要素：人物数量(1girl/2boys)、角色五官/瞳色/发色发型/表情/服装/配饰/动作，以及背景物体。\n"
                "2. 构图：构图与取景(upper body, cowboy shot, from above)、景别、视角。\n"
                "3. 光影：光源与光线(soft lighting, backlighting, rim light, dramatic lighting)。\n"
                "4. 风格：画风(anime, illustration, masterpiece, best quality)、色彩。\n"
                "要求：输出逗号分隔英文 danbooru 标签，按重要性排序；只标可见内容，不解释。\n"
                "- 【直白打标】若含裸露/性内容，必须直接打出对应的 danbooru 标签"
                "（nude、nipples、penis、pussy、vaginal、spreading、cum 等），不得用 generic/suggestive/intimate 等回避词代替。"
                "标签是给 AI 用的，含糊会丢失语义。"
            ),
            "description": "二次元向 danbooru 标签",
        },
        "📝 自然语言描述": {
            "system_prompt": (
                "你是图像描述专家。请用一段连贯的英文自然语言描述这张图片，依次覆盖：\n"
                "1. 画面要素：主体与物体的具体特征。\n"
                "2. 构图：取景、视角、景别、主体位置。\n"
                "3. 光影：光源、方向、光质、阴影与色调。\n"
                "4. 风格：艺术风格、色彩与质感。\n"
                "要求：客观准确，仅描述图中可见内容，输出一段英文，不使用列表/编号。"
            ),
            "description": "自然语言段落描述",
        },
        "🔄 仅内容（风格迁移用）": {
            "system_prompt": (
                "你是一位图像内容描述专家，任务是产出「仅描述画面内容」的英文描述，用于后续的风格转换/重绘。\n"
                "\n"
                "输出必须以自然语言为主、连续成段，danbooru 风格的短标签最多只能作为辅助穿插，不能整段都是标签。\n"
                "\n"
                "请按下面的空间结构来写，顺序固定：\n"
                "1. 画面中央是谁/什么：中央主体的身份、数量（如 a young woman / two children / a black bird）。\n"
                "2. 主体长什么样：五官、肤色、发色发型、表情、体型。\n"
                "3. 穿着打扮：服装款式与颜色、鞋帽、配饰、首饰、随身道具。\n"
                "4. 在做什么：动作、姿态、手势、视线朝向、与其他对象的互动。\n"
                "5. 画面边缘有什么：四周边缘的物体、点缀元素、前景遮挡物。\n"
                "6. 背景是什么：背景环境、场景、远处景物、地平面/天空/墙面等。\n"
                "（构图信息如主体位置、视角、景别可以自然地融进上述描述里，不必单列。）\n"
                "\n"
                "【重要：风格中立描述】描述必须保持客观、克制，避免任何会把生成模型带偏到特定风格（尤其是二次元/动漫）的措辞：\n"
                "- 不要用夸张的程度/尺寸副词或形容词，例如 large、big、huge、prominent、striking、massive、enormous、wide、giant。\n"
                "  （错误：large blue eyes, prominent blush；正确：blue eyes, a flush on her cheeks）\n"
                "- 只描述客观颜色与位置，不描述程度：五官/表情直接写事实（has blue eyes, freckles across the nose），"
                "不要加 large/sparkling/exaggerated/prominent 之类修饰。\n"
                "- 不要用动漫化的情绪组合词（如 sparkling eyes、teary-eyed smile、blushing smile），"
                "改成中性事实描述（tears at the corners of her eyes, a flush on the cheeks）。\n"
                "- 描述特征时用「事实」而非「印象」：写「a flush on her cheeks」而不是「a prominent blush」；"
                "写「wavy black hair reaching her shoulders」而不是「long flowing black hair」。\n"
                "\n"
                "【严格禁止】以下内容绝对不能出现在输出中（这些要留给重绘时的目标风格自由发挥）：\n"
                "- 任何光影相关：光源、光照、光质、阴影、明暗、对比度、氛围光、golden hour、lighting 等词汇。\n"
                "- 任何风格相关：写实/照片/插画/二次元/油画/水彩/3D/赛博朋克/画师名/作品名等，"
                "以及 masterpiece、best quality、render、photorealistic、illustration 等画质/风格标签。\n"
                "- 任何画质/相机/胶片相关：8k、ultra detailed、DSLR、film、bokeh、sharp focus 等。\n"
                "\n"
                "要求：\n"
                "- 用英文写成一段连贯的自然语言，按上面 1→6 的顺序；可用少量括号标签辅助，但以句子为主。\n"
                "- 客观描述图中可见的内容，不编造图中没有的东西。\n"
                "- 检查并删除任何夸张修饰词（large/prominent 等）和任何光影或风格词。\n"
                "- 【直白描述】若画面含裸露、性、暴力、血腥等内容，必须用直白、准确、解剖学术语直接描述"
                "（如 nude、bare breasts、penis、vulva、spreading legs、vaginal penetration、blood、open wound），"
                "不得用「intimate」「revealing」「suggestive」「graphic」「explicit」等含糊回避词代替。"
                "这是给风格迁移用的内容描述，含糊会丢失语义导致无法复现。"
            ),
            "description": "自然语言为主的画面内容描述（中央主体→外观→服装→动作→边缘→背景），剥离光影与风格，供风格转换/重绘使用",
        },
        "📷 强制真实化（一律描述成照片）": {
            "system_prompt": (
                "你是一位写实摄影图像描述专家。无论原图是什么风格（写实照片、二次元插画、油画、3D、概念图等），"
                "你都必须把它描述成一张真实拍摄的照片——假设这就是一张抓拍/摆拍的真实现场照片，用写实的摄影语言描述。\n"
                "\n"
                "输出以自然语言为主、连续成段，danbooru 风格的短标签最多只能作为辅助穿插。\n"
                "\n"
                "【核心规则：一切虚构特征都当成真实存在的物理道具/妆容/造型】\n"
                "- 动物耳/兽耳 → 描述成真实的 cosplay 头饰、毛绒发箍或头戴配件（如 'a pair of pointed fuzzy headbands resembling animal ears'）。\n"
                "- 光环/悬浮物 → 描述成真实的道具，如悬挂/支撑的圆环、LED 光环道具、头饰配件（如 'a pink ring-shaped prop suspended above her head'）。\n"
                "- 不可能的发色（蓝/粉/紫/红/绿等） → 描述成染色头发（'color-dyed hair'），并补 'visible hair dye, dyed strands'。\n"
                "- 不可能的瞳色（红/紫等） → 描述成彩色美瞳（'colored contact lenses'）。\n"
                "- 魔法/特效/发光 → 描述成真实的灯光/烟雾/反光（如舞台灯、烟雾机、玻璃反光）。\n"
                "- 虚构生物/科幻元素 → 描述成 cosplay 服装、特效化妆、道具或布景。\n"
                "\n"
                "【必须使用写实摄影语言】描述人物和场景时强制带上这些照片语料里的特征：\n"
                "- 皮肤：natural skin texture\n"
                "- 面部：asymmetrical facial features, fine lines, realistic facial proportions\n"
                "- 材质：fabric weave texture, stitching, scuffs and wear on props\n"
                "- 摄影：captured on a camera, candid, in-camera, raw photograph\n"
                "（把上述合适的特征自然融进描述里，不要简单堆砌）\n"
                "\n"
                "【严格禁止】任何会把生成模型带回绘画/动漫的词：\n"
                "- 禁止：illustration、anime、manga、painting、oil painting、watercolor、concept art、"
                "cel shading、3d render、cgi、vector art、digital art、fanart\n"
                "- 禁止：masterpiece、best quality、highly detailed（这类是绘画/动漫区的画质标签）\n"
                "- 禁止描述成「画风」：不要说 'the style of...'、'reminiscent of...画师'。\n"
                "\n"
                "请按这个顺序组织描述：\n"
                "1. 中央主体是谁/什么（按真实人/物来写）。\n"
                "2. 外观：五官（按真人写）、肤色、发色（按染发写）、体型、皮肤质感。\n"
                "3. 穿着与道具：服装、鞋帽、配饰，以及所有虚构特征对应的真实道具/妆造。\n"
                "4. 动作/姿态。\n"
                "5. 画面边缘与前景。\n"
                "6. 背景环境。\n"
                "\n"
                "要求：用英文写成一段连贯的自然语言，客观描述图中内容（把虚构特征如实转译为真实道具），"
                "确保整段读起来像在描述一张真实照片。\n"
                "- 【直白描述】若画面含裸露、性、暴力、血腥等内容，按真实人体/真实事件直白描述"
                "（如 nude、bare breasts、penis、vulva、spreading、penetration、blood、wound），"
                "不得用 intimate、revealing、suggestive、graphic、explicit 等含糊回避词代替。"
                "照片描述必须准确，含糊会丢失语义。"
            ),
            "description": "无论原图什么风格，一律描述成真实照片（虚构特征转译为道具/染发/美瞳，并强制写实皮肤/摄影语言）",
        },
    }

    @classmethod
    def get_preset_names(cls):
        return list(cls.PRESETS.keys())

    @classmethod
    def get_endpoints(cls):
        """与 PromptEnhancer 共用同一份 endpoint 记录。"""
        records = _load_prompt_enhance_records()
        names = list(records.keys())
        if not names:
            return ["(无记录，请先用设置节点添加)"]
        return names

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "endpoint": (cls.get_endpoints(),),
                "preset": (cls.get_preset_names(), {"default": "🔎 通用描述（推荐）"}),
                "model_id": ("STRING", {"default": "", "placeholder": "留空则使用记录中的默认模型（需为支持视觉的模型）"}),
                "system_prompt": ("STRING", {"default": "", "multiline": True, "placeholder": "系统提示词（选了预设则叠加在预设之后；选\"不使用预设\"则单独生效）"}),
                "extra_prompt": ("STRING", {"default": "", "multiline": True, "placeholder": "附加用户指令（可选，如\"重点描述人物服装\"）"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "temperature": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.05}),
                "max_tokens": ("INT", {"default": 1024, "min": 1, "max": 32768, "step": 1}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, image, endpoint, preset, model_id, system_prompt, extra_prompt, seed, **kwargs):
        # 确定性缓存键：图像内容指纹 + 端点记录 + 预设 + 提示词 + seed。
        # 全部未变 → 复用上次结果，跳过重复的 Vision 请求。
        import hashlib
        records = _load_prompt_enhance_records()
        rec_blob = json.dumps(records.get(endpoint, {}), ensure_ascii=False, sort_keys=True)
        # batch：把每张图的指纹串起来
        if image is not None and hasattr(image, "shape") and image.dim() >= 4:
            img_hashes = ",".join(_hash_image_tensor(image[i]) for i in range(image.shape[0]))
        else:
            img_hashes = "no_image"
        key = "|".join([
            img_hashes,
            rec_blob,
            str(preset),
            str(model_id or ""),
            str(system_prompt or ""),
            str(extra_prompt or ""),
            str(seed),
            str(kwargs.get("temperature", 0.5)),
            str(kwargs.get("max_tokens", 1024)),
        ])
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("description", "status")
    FUNCTION = "analyze"
    CATEGORY = "text"

    def analyze(self, image, endpoint, preset, model_id, system_prompt, extra_prompt, seed, temperature=0.5, max_tokens=1024):
        import requests

        if image is None:
            return ("", "❌ 未输入图像")

        records = _load_prompt_enhance_records()
        if endpoint not in records:
            return ("", f"❌ 找不到记录「{endpoint}」，请先用设置节点添加")

        rec = records[endpoint]
        base_url = rec.get("url", "").strip()
        api_key = rec.get("api_key", "")
        use_model = (model_id or rec.get("model_id", "")).strip()

        if not use_model:
            return ("", "❌ 未指定模型 id（请填写支持视觉的模型，如 gpt-4o / doubao-vision）")
        if not base_url:
            return ("", f"❌ 记录「{endpoint}」缺少 url")

        # 合成最终系统提示词：预设在前，自定义在后
        preset_sp = self.PRESETS.get(preset, {}).get("system_prompt", "")
        custom_sp = (system_prompt or "").strip()
        if preset_sp and custom_sp:
            effective_sp = preset_sp + "\n\n" + custom_sp
        else:
            effective_sp = preset_sp or custom_sp

        chat_url = _join_openai_path(base_url, "chat/completions")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        # 处理 batch：N 张图分别描述
        n = image.shape[0] if hasattr(image, "shape") and image.dim() >= 4 else 1
        descriptions = []
        total_tokens = 0
        for i in range(n):
            # 统一取单张 [H,W,C]；_image_to_data_url 内部对 4D 也有兜底
            img_tensor = image[i] if hasattr(image, "shape") and image.dim() >= 4 else image
            data_url = _image_to_data_url(img_tensor)

            # OpenAI Vision 标准格式：content 为多模态块数组
            content = [
                {"type": "text", "text": (f"[seed={seed}] " + (extra_prompt + " " if extra_prompt else "")).strip()},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]
            messages = []
            if effective_sp:
                messages.append({"role": "system", "content": effective_sp})
            messages.append({"role": "user", "content": content})

            payload = {
                "model": use_model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }

            try:
                resp = requests.post(chat_url, headers=headers, json=payload, timeout=120)
            except requests.exceptions.Timeout:
                return ("", f"❌ 第 {i+1}/{n} 张图请求超时")
            except Exception as e:
                return ("", f"❌ 第 {i+1}/{n} 张图请求异常：{e}")

            if resp.status_code != 200:
                return ("", f"❌ 第 {i+1}/{n} 张图 HTTP {resp.status_code}: {resp.text[:300]}")

            try:
                data = resp.json()
                content_text = _extract_content(data)
            except Exception as e:
                return ("", f"❌ 第 {i+1}/{n} 张图解析响应失败：{e}\n原始：{resp.text[:300]}")

            descriptions.append(content_text)
            total_tokens += data.get("usage", {}).get("total_tokens", 0)

        # 多张图用分隔符分开；单张直接输出
        if n > 1:
            description = "\n\n---\n\n".join(descriptions)
        else:
            description = descriptions[0] if descriptions else ""

        preset_tag = preset if preset != "(不使用预设)" else "自定义"
        status = (f"✅ 成功 | 端点: {endpoint} | 模型: {use_model} | 预设: {preset_tag} | "
                  f"图片数: {n} | seed: {seed} | tokens: {total_tokens}")
        return (description, status)


NODE_CLASS_MAPPINGS.update({
    "PromptEnhanceSettings": PromptEnhanceSettings,
    "PromptEnhancer": PromptEnhancer,
    "ImageAnalyzer": ImageAnalyzer,
})

NODE_DISPLAY_NAME_MAPPINGS.update({
    "PromptEnhanceSettings": "🤖 提示词扩写设置 (Prompt Enhance Settings)",
    "PromptEnhancer": "🤖 提示词扩写 (Prompt Enhancer)",
    "ImageAnalyzer": "🖼️ 图片分析 (Image Analyzer)",
})


