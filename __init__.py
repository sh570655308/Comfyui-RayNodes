from .ray_nodes import *

# Node class mapping
NODE_CLASS_MAPPINGS = {
    "BracketedTagIndexMerger": BracketedTagIndexMerger,
    "Florence2TagProcessor": Florence2TagProcessor,
    "ImageListConverter": ImageListConverter,
    "ImageSelector": ImageSelector,
    "MaskBlackener": MaskBlackener,
    "MaskApplierAndCombiner": MaskApplierAndCombiner,
    "MaskProcessor": MaskProcessor,
    "TagArrayToLines": TagArrayToLines,
    "TagIndexMerger": TagIndexMerger,
    "GrabberTagProcessor": GrabberTagProcessor,
    "ImageResizer": ImageResizer,
    "SaveImageWebsocket": SaveImageWebsocket,
    "BorderMask": BorderMask,
    "SaturationAdjuster": SaturationAdjuster,
}

# Node display name mapping
NODE_DISPLAY_NAME_MAPPINGS = {
    "BracketedTagIndexMerger": "🏷️ Bracketed Tag-Index Merger",
    "Florence2TagProcessor": "🏷️ Florence2 Tag Processor",
    "ImageListConverter": "🖼️Image List Converter",
    "ImageSelector": "🖼️ Image Selector",
    "MaskBlackener": "🖤Mask Blackener",
    "MaskApplierAndCombiner": "🎭Mask Applier and Combiner",
    "MaskProcessor": "🎭Mask Processor",
    "TagArrayToLines": "📄 Tag Array to Lines",
    "TagIndexMerger": "🏷️ Tag-Index Merger",
    "GrabberTagProcessor": "Grabber Tag Processor",
    "ImageResizer": "🖼️Image Resizer",
    "SaveImageWebsocket": "Save Image Websocket",
    "BorderMask": "🎭Border Mask",
    "SaturationAdjuster": "🌈SaturationAdjuster",
}
