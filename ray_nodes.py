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

NODE_CLASS_MAPPINGS = {
    "Florence2TagProcessor": Florence2TagProcessor
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Florence2TagProcessor": "🏷️ Florence2 Tag Processor"
}



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
                elif isinstance(image, List):  # List of images
                    for img in image:
                        if isinstance(img, torch.Tensor) and img.dim() == 3:
                            image_list.append(img.unsqueeze(0))

        if not image_list:
            # Return an empty tensor if no valid images were found
            return (torch.empty((0, 3, 1, 1)),)

        # Stack all images into a single tensor
        return (torch.cat(image_list, dim=0),)

NODE_CLASS_MAPPINGS = {
    "ImageListConverter": ImageListConverter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageListConverter": "🖼️Image List Converter"
}



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

NODE_CLASS_MAPPINGS = {
    "ImageSelector": ImageSelector
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageSelector": "🖼️ Image Selector"
}



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

NODE_CLASS_MAPPINGS = {
    "MaskBlackener": MaskBlackener,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MaskBlackener": "🖤Mask Blackener"
}



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
    """Apply feathering effect to a mask."""
    if feather_amount <= 0:
        return mask_np
    
    # 确保mask是灰度图
    if len(mask_np.shape) > 2:
        mask_np = cv2.cvtColor(mask_np, cv2.COLOR_RGB2GRAY)
    
    # 转换为float32类型以保持精度
    mask_np = mask_np.astype(np.float32) / 255.0

    # 使用距离变换来创建渐变
    dist_transform = cv2.distanceTransform((mask_np * 255).astype(np.uint8), cv2.DIST_L2, 3)
    dist_transform = cv2.normalize(dist_transform, None, 0, 1, cv2.NORM_MINMAX)
    
    # 创建高斯核进行模糊
    kernel_size = feather_amount * 2 + 1  # 确保核大小为奇数
    sigma = feather_amount / 2
    blurred = cv2.GaussianBlur(mask_np, (kernel_size, kernel_size), sigma)

    # 在边缘区域混合原始mask和模糊后的mask
    edge_kernel = np.ones((3, 3), np.uint8)
    edges = cv2.Canny((mask_np * 255).astype(np.uint8), 100, 200)
    dilated_edges = cv2.dilate(edges, edge_kernel, iterations=feather_amount)
    edge_mask = dilated_edges.astype(np.float32) / 255.0

    # 使用边缘mask来混合原始mask和模糊后的mask
    result = np.where(edge_mask > 0, blurred, mask_np)
    
    # 重新缩放到0-255范围
    result = np.clip(result * 255, 0, 255)
    
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

NODE_CLASS_MAPPINGS = {
    "MaskApplierAndCombiner": MaskApplierAndCombiner,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MaskApplierAndCombiner": "🎭Mask Applier and Combiner"
}



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
        检查mask是否有效（是否包含至少两种不同的值）
        增加调试信息输出
        """
        # 将tensor转换为numpy数组以便处理
        mask_np = mask.cpu().numpy()
        
        # 打印调试信息
        min_val = np.min(mask_np)
        max_val = np.max(mask_np)
        print(f"Mask value range: min={min_val}, max={max_val}")
        
        # 获取唯一值（使用更小的精度以处理浮点误差）
        unique_values = np.unique(np.round(mask_np, decimals=1))
        print(f"Unique values in mask: {unique_values}")
        
        # 检查是否包含0和1（允许一定的误差范围）
        has_zero = np.any(np.abs(mask_np) < 0.1)
        has_one = np.any(np.abs(mask_np - 1) < 0.1)
        
        is_valid = has_zero and has_one
        print(f"Has zero: {has_zero}, Has one: {has_one}, Is valid: {is_valid}")
        
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
                # 检查mask是否有效（包含两种不同的值）
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
NODE_CLASS_MAPPINGS = {
    "MaskProcessor": MaskProcessor
}

# 节点显示名称映射
NODE_DISPLAY_NAME_MAPPINGS = {
    "MaskProcessor": "🎭Mask Processor"
}



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

NODE_CLASS_MAPPINGS = {
    "TagArrayToLines": TagArrayToLines
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TagArrayToLines": "📄 Tag Array to Lines"
}



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

NODE_CLASS_MAPPINGS = {
    "TagIndexMerger": TagIndexMerger
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TagIndexMerger": "🏷️ Tag-Index Merger"
}



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

NODE_CLASS_MAPPINGS = {
    "GrabberTagProcessor": GrabberTagProcessor
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GrabberTagProcessor": "Grabber Tag Processor"
}



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
            },
            "optional": {}
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("resized_image",)
    FUNCTION = 'resize_image'
    CATEGORY = 'Image Processing'

    def resize_image(self, image_a, image_b):
        def process_single_image(i_a, i_b):
            # Convert tensors to PIL for processing
            pil_a = tensor2pil(i_a.unsqueeze(0) if i_a.dim() == 3 else i_a).convert('RGB')
            pil_b = tensor2pil(i_b.unsqueeze(0) if i_b.dim() == 3 else i_b).convert('RGB')
            
            # Get target size from image_b
            target_size = pil_b.size
            
            # Resize image_a to the target size using Lanczos algorithm
            resized_pil_a = pil_a.resize(target_size, Image.LANCZOS)
            
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

NODE_CLASS_MAPPINGS = {
    "ImageResizer": ImageResizer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageResizer": "🖼️Image Resizer"
}



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

NODE_CLASS_MAPPINGS = {
    "SaveImageWebsocket": SaveImageWebsocket,
}

