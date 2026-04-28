import argparse
import json
import math
import os
import types
from collections import defaultdict

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from rfdetr import RFDETRMedium

transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


def load_image_as_tensor(image_path, transform, device):
    img_pil = Image.open(image_path).convert("RGB")
    img_tensor = transform(img_pil).unsqueeze(0).to(device)   # [1, 3, 640, 640]
    return img_pil, img_tensor


def coco_xywh_to_rel_cxcywh(coco_bbox, image_width, image_height):
    x, y, w, h = coco_bbox

    cx = x + w / 2.0
    cy = y + h / 2.0

    return [
        cx / image_width,
        cy / image_height,
        w / image_width,
        h / image_height,
    ]


def get_gt_boxes_for_image(image_id, images_by_id, anns_by_image_id, category_id=None):
    """
    returns:
        gt_boxes_rel: Liste von [cx, cy, w, h] relativ [0,1]
        gt_ann_list : originale Annotationen (optional nützlich fürs Debugging)
    """
    img_info = images_by_id[image_id]
    W = img_info["width"]
    H = img_info["height"]

    anns = anns_by_image_id.get(image_id, [])

    if category_id is not None:
        anns = [ann for ann in anns if ann["category_id"] == category_id]

    gt_boxes_rel = [
        coco_xywh_to_rel_cxcywh(ann["bbox"], W, H)
        for ann in anns
    ]

    return gt_boxes_rel, anns


def normalize_cam(cam, mode="negative", eps=1e-8):
    """
    cam: [H, W] tensor

    modes:
      - 'signed'   : normalize to approx [-1, 1]
      - 'relu'     : positive evidence only
      - 'negative' : negative evidence only
      - 'minmax'   : plain min-max
      - 'abs'      : absolute magnitude
    """
    cam = cam.detach()

    if mode == "signed":
        return cam / (cam.abs().max() + eps)

    if mode == "relu":
        cam = torch.relu(cam)
        return cam / (cam.max() + eps)

    if mode == "negative":
        cam = torch.relu(-cam)
        return cam / (cam.max() + eps)

    if mode == "minmax":
        cam = cam - cam.min()
        return cam / (cam.max() + eps)

    if mode == "abs":
        cam = cam.abs()
        return cam / (cam.max() + eps)

    raise ValueError(f"Unknown normalization mode: {mode}")


def box_cxcywh_to_xyxy_pixel(box, pil_img):
    """
    box: [cx, cy, w, h] normalized to [0,1]
    returns: [x0, y0, x1, y1] in pixel coordinates
    """
    w_img, h_img = pil_img.size
    cx, cy, w, h = box.tolist()

    x0 = (cx - w / 2) * w_img
    y0 = (cy - h / 2) * h_img
    x1 = (cx + w / 2) * w_img
    y1 = (cy + h / 2) * h_img

    return [x0, y0, x1, y1]


def resize_cam_to_image(cam, pil_img):
    """
    cam: [H, W] tensor or numpy array
    returns resized cam as numpy array
    """
    if torch.is_tensor(cam):
        cam_t = cam.detach().cpu().float()
    else:
        cam_t = torch.tensor(cam, dtype=torch.float32)

    cam_up = F.interpolate(
        cam_t.unsqueeze(0).unsqueeze(0),
        size=(pil_img.size[1], pil_img.size[0]),
        mode="bilinear",
        align_corners=False,
    )[0, 0].numpy()

    return cam_up


def show_cam_on_image(
    pil_img,
    cam,
    title="Grad-CAM",
    cmap="jet",
    alpha=0.4,
    box_xyxy=None,
):
    """
    Display CAM overlay on original image.
    """
    img_np = np.array(pil_img).astype(np.float32) / 255.0
    cam_up = resize_cam_to_image(cam, pil_img)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(img_np)
    ax.imshow(cam_up, cmap=cmap, alpha=alpha)

    if box_xyxy is not None:
        x0, y0, x1, y1 = box_xyxy
        rect = plt.Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            linewidth=2,
            edgecolor="lime",
            facecolor="none",
        )
        ax.add_patch(rect)

    ax.set_title(title)
    ax.axis("off")
    plt.show()

    return cam_up


def get_conv_gradcam(
    core_model,
    img_tensor,
    pil_img,
    target_layer=None,
    target_class=0,
    target_query=None,
    query_strategy="top_score",
    cam_mode="negative",
    use_box_term=False,
    box_weight=0.5,
    show=True,
    draw_box=False,
):
    """
    Conv-based Grad-CAM for RF-DETR backbone/projector layers.

    Returns dict with:
      - cam
      - raw_cam
      - cam_up
      - target_query
      - target_class
      - target_score
      - target_logit
      - target_box
      - target_value
      - pred_logits
      - pred_boxes
    """
    if target_layer is None:
        target_layer = core_model.backbone[0].projector.stages[0][0].cv2.conv

    activations = {}
    gradients = {}

    def forward_hook(module, inp, out):
        activations["value"] = out

    def backward_hook(module, grad_input, grad_output):
        gradients["value"] = grad_output[0]

    fwd_handle = target_layer.register_forward_hook(forward_hook)
    bwd_handle = target_layer.register_full_backward_hook(backward_hook)

    core_model.zero_grad()
    outputs = core_model(img_tensor)

    pred_logits = outputs["pred_logits"]   # [1, 300, num_classes]
    pred_boxes = outputs["pred_boxes"]     # [1, 300, 4]

    if target_query is None:
        target_query, target_score = select_target_query(
            pred_logits,
            target_class=target_class,
            strategy=query_strategy,
        )
    else:
        probs = pred_logits[0].softmax(-1)
        target_score = probs[target_query, target_class].item()

    target_logit = pred_logits[0, target_query, target_class]
    target_box = pred_boxes[0, target_query]

    if use_box_term:
        target = target_logit + box_weight * target_box.sum()
    else:
        target = target_logit

    target.backward()

    act = activations["value"]   # [1, C, H, W]
    grad = gradients["value"]    # [1, C, H, W]

    fwd_handle.remove()
    bwd_handle.remove()

    weights = grad.mean(dim=(2, 3), keepdim=True)         # [1, C, 1, 1]
    raw_cam = (weights * act).sum(dim=1, keepdim=True)    # [1, 1, H, W]
    raw_cam_2d = raw_cam[0, 0].detach().cpu()

    cam = normalize_cam(raw_cam_2d, mode=cam_mode)

    cam_up = resize_cam_to_image(cam, pil_img)

    if show:
        cmap = "seismic" if cam_mode == "signed" else "jet"
        title_extra = "class+box" if use_box_term else "class-only"

        box_xyxy = None
        if draw_box:
            box_xyxy = box_cxcywh_to_xyxy_pixel(target_box.detach().cpu(), pil_img)

        show_cam_on_image(
            pil_img=pil_img,
            cam=cam,
            title=f"Conv Grad-CAM | q={target_query} c={target_class} | {cam_mode} | {title_extra}",
            cmap=cmap,
            alpha=0.4,
            box_xyxy=box_xyxy,
        )

    return {
        "cam": cam.detach().cpu(),
        "raw_cam": raw_cam_2d.detach().cpu(),
        "cam_up": cam_up,
        "target_query": target_query,
        "target_class": target_class,
        "target_score": target_score,
        "target_logit": target_logit.item(),
        "target_box": target_box.detach().cpu(),
        "target_value": target.item(),
        "pred_logits": pred_logits.detach().cpu(),
        "pred_boxes": pred_boxes.detach().cpu(),
    }


def select_target_query(pred_logits, target_class=0, strategy="top_score"):
    probs = pred_logits[0].softmax(-1)

    if strategy == "top_score":
        scores = probs[:, target_class]
        top_scores, top_idx = scores.sort(descending=True)
        target_query = top_idx[0].item()
        target_score = top_scores[0].item()
        return target_query, target_score

    raise ValueError(f"Unknown strategy: {strategy}")


def patch_decoder_self_attn_to_store_weights(core_model):
    patched = []

    for layer in core_model.transformer.decoder.layers:
        mha = layer.self_attn
        orig_forward = mha.forward

        def wrapped_forward(self, *args, _orig_forward=orig_forward, **kwargs):
            kwargs = dict(kwargs)
            kwargs["need_weights"] = True
            kwargs["average_attn_weights"] = False

            out = _orig_forward(*args, **kwargs)

            if isinstance(out, tuple) and len(out) >= 2:
                attn_output, attn_weights = out[0], out[1]
                self._last_attn_weights = attn_weights
                return attn_output, attn_weights

            self._last_attn_weights = None
            return out

        mha._orig_forward_for_rollout = orig_forward
        mha.forward = types.MethodType(wrapped_forward, mha)
        mha._last_attn_weights = None
        patched.append(mha)

    def restore_fn():
        for mha in patched:
            if hasattr(mha, "_orig_forward_for_rollout"):
                mha.forward = mha._orig_forward_for_rollout
                del mha._orig_forward_for_rollout
            if hasattr(mha, "_last_attn_weights"):
                del mha._last_attn_weights

    return restore_fn


def _normalize_attn_with_residual(attn, add_residual=True, eps=1e-8):
    q = attn.shape[0]
    if add_residual:
        attn = attn + torch.eye(q, device=attn.device, dtype=attn.dtype)
    attn = attn / (attn.sum(dim=-1, keepdim=True) + eps)
    return attn


def get_decoder_self_attn_rollout(
    core_model,
    img_tensor,
    target_class=0,
    target_query=None,
    query_strategy="top_score",
    add_residual=True,
):
    restore_fn = patch_decoder_self_attn_to_store_weights(core_model)

    try:
        core_model.zero_grad()
        outputs = core_model(img_tensor)

        pred_logits = outputs["pred_logits"]
        pred_boxes = outputs["pred_boxes"]

        if target_query is None:
            target_query, target_score = select_target_query(
                pred_logits,
                target_class=target_class,
                strategy=query_strategy,
            )
        else:
            probs = pred_logits[0].softmax(-1)
            target_score = probs[target_query, target_class].item()

        target_logit = pred_logits[0, target_query, target_class]

        self_attn_maps = []
        for i, layer in enumerate(core_model.transformer.decoder.layers):
            w = getattr(layer.self_attn, "_last_attn_weights", None)
            if w is None:
                raise RuntimeError(f"Missing self-attn weights in decoder layer {i}")

            # [B, heads, Q, Q] -> [Q, Q]
            w = w[0].mean(dim=0)
            self_attn_maps.append(w.detach())

    finally:
        restore_fn()

    q = self_attn_maps[0].shape[0]
    rollout = torch.eye(q, device=self_attn_maps[0].device, dtype=self_attn_maps[0].dtype)

    for attn in self_attn_maps:
        a = _normalize_attn_with_residual(attn, add_residual=add_residual)
        rollout = a @ rollout

    query_rollout = rollout[target_query]
    query_rollout = query_rollout / (query_rollout.sum() + 1e-8)

    return {
        "target_query": target_query,
        "target_class": target_class,
        "target_score": target_score,
        "target_logit": target_logit.item(),
        "query_rollout": query_rollout.detach().cpu(),
        "rollout_matrix": rollout.detach().cpu(),
        "self_attn_maps": [x.detach().cpu() for x in self_attn_maps],
        "pred_logits": pred_logits.detach().cpu(),
        "pred_boxes": pred_boxes.detach().cpu(),
    }


def _infer_num_heads_num_points(module, num_levels):
    num_heads = getattr(module, "n_heads", None)
    if num_heads is None:
        num_heads = getattr(module, "num_heads", None)
    if num_heads is None:
        num_heads = getattr(module, "n_head", None)

    num_points = getattr(module, "n_points", None)
    if num_points is None:
        num_points = getattr(module, "num_points", None)

    if num_heads is not None and num_points is not None:
        return int(num_heads), int(num_points)

    out_dim_offsets = module.sampling_offsets.out_features
    out_dim_weights = module.attention_weights.out_features

    # out_dim_offsets = heads * levels * points * 2
    # out_dim_weights = heads * levels * points
    found = []
    for h in [1, 2, 4, 8, 16, 32]:
        if out_dim_weights % (h * num_levels) == 0:
            p = out_dim_weights // (h * num_levels)
            if out_dim_offsets == h * num_levels * p * 2:
                found.append((h, p))

    if len(found) == 0:
        raise RuntimeError("Could not infer num_heads / num_points from MSDeformAttn module.")

    # meistens 8 heads
    return found[-1]


def _compute_sampling_locations(reference_points, sampling_offsets, num_points):
    """
    reference_points: [B,Q,L,4]
    sampling_offsets: [B,Q,H,L,P,2]
    returns:
      sampling_locations: [B,Q,H,L,P,2] in normalized coords
    """
    if reference_points.shape[-1] != 4:
        raise ValueError(f"Expected reference_points last dim = 4, got {reference_points.shape[-1]}")

    ref_xy = reference_points[..., :2][:, :, None, :, None, :]   # [B,Q,1,L,1,2]
    ref_wh = reference_points[..., 2:][:, :, None, :, None, :]   # [B,Q,1,L,1,2]

    sampling_locations = ref_xy + sampling_offsets / float(num_points) * ref_wh * 0.5
    return sampling_locations


def _splat_points_to_map(weights, locations, h, w):
    """
    weights:   [P]
    locations: [P,2] normalized x,y
    returns:
      map: [H,W]
    """
    device = weights.device
    dtype = weights.dtype

    out = torch.zeros((h, w), device=device, dtype=dtype)

    xs = locations[:, 0] * (w - 1)
    ys = locations[:, 1] * (h - 1)

    x0 = torch.floor(xs).long().clamp(0, w - 1)
    x1 = (x0 + 1).clamp(0, w - 1)
    y0 = torch.floor(ys).long().clamp(0, h - 1)
    y1 = (y0 + 1).clamp(0, h - 1)

    dx = xs - x0.float()
    dy = ys - y0.float()

    wa = (1 - dx) * (1 - dy)
    wb = dx * (1 - dy)
    wc = (1 - dx) * dy
    wd = dx * dy

    out.index_put_((y0, x0), weights * wa, accumulate=True)
    out.index_put_((y0, x1), weights * wb, accumulate=True)
    out.index_put_((y1, x0), weights * wc, accumulate=True)
    out.index_put_((y1, x1), weights * wd, accumulate=True)

    return out


def _project_msdeform_query_to_map(attn_w_q, samp_locs_q, spatial_shapes, out_h, out_w):
    """
    attn_w_q:    [heads, L, P]
    samp_locs_q: [heads, L, P, 2]
    spatial_shapes: [L,2] with (H_l, W_l)

    returns:
      dense map [out_h, out_w]
    """
    device = attn_w_q.device
    dtype = attn_w_q.dtype

    num_heads, num_levels, num_points = attn_w_q.shape
    dense = torch.zeros((out_h, out_w), device=device, dtype=dtype)

    for lvl in range(num_levels):
        h_l = int(spatial_shapes[lvl, 0].item())
        w_l = int(spatial_shapes[lvl, 1].item())

        lvl_map = torch.zeros((h_l, w_l), device=device, dtype=dtype)

        for head in range(num_heads):
            lvl_map += _splat_points_to_map(
                weights=attn_w_q[head, lvl],
                locations=samp_locs_q[head, lvl],
                h=h_l,
                w=w_l,
            )

        lvl_map_up = F.interpolate(
            lvl_map[None, None],
            size=(out_h, out_w),
            mode="bilinear",
            align_corners=False,
        )[0, 0]

        dense += lvl_map_up

    return dense


def get_decoder_self_cross_rollout_map(
    core_model,
    img_tensor,
    target_class=0,
    target_query=None,
    query_strategy="top_score",
    cam_mode="relu",
    add_self_attn_residual=True,
):
    """
    Kombiniert:
      - decoder self-attn rollout (Query -> Query)
      - decoder deformable cross-attn projection (Query -> Bild)

    Rückgabe:
      - rollout_map: [H_img, W_img]
      - raw_rollout_map: [H_img, W_img]
      - query_rollout: [Q]
      - per_query_map: [Q, H_img, W_img]
      - target_query, ...
    """
    device = img_tensor.device
    img_h, img_w = int(img_tensor.shape[-2]), int(img_tensor.shape[-1])

    # 1) self-attn rollout
    self_result = get_decoder_self_attn_rollout(
        core_model=core_model,
        img_tensor=img_tensor,
        target_class=target_class,
        target_query=target_query,
        query_strategy=query_strategy,
        add_residual=add_self_attn_residual,
    )

    target_query = self_result["target_query"]
    query_rollout = self_result["query_rollout"].to(device)   # [Q]

    # 2) cross-attn hook
    saved_cross = []
    handles = []

    def make_cross_hook(layer_idx):
        def cross_hook(module, inp, out):
            # laut deinem Debug:
            # inp[0] query             [B,Q,C]
            # inp[1] reference_points  [B,Q,L,4]
            # inp[2] value             [B,SumHW,C]
            # inp[3] spatial_shapes    [L,2]
            # inp[4] level_start_index [L]
            # inp[5] padding_mask      [B,SumHW]

            record = {"layer": layer_idx}

            query = inp[0]
            reference_points = inp[1]
            value = inp[2]
            spatial_shapes = inp[3]
            level_start_index = inp[4]
            padding_mask = inp[5] if len(inp) > 5 else None

            # falls spatial_shapes zufällig [1,2] ist, ist das schon [L,2]
            if spatial_shapes.dim() != 2 or spatial_shapes.shape[-1] != 2:
                raise RuntimeError(f"Unexpected spatial_shapes shape: {tuple(spatial_shapes.shape)}")

            b, q, c = query.shape
            num_levels = spatial_shapes.shape[0]

            num_heads, num_points = _infer_num_heads_num_points(module, num_levels)

            sampling_offsets = module.sampling_offsets(query)
            attention_weights = module.attention_weights(query)

            sampling_offsets = sampling_offsets.view(
                b, q, num_heads, num_levels, num_points, 2
            )

            attention_weights = attention_weights.view(
                b, q, num_heads, num_levels, num_points
            )
            attention_weights = F.softmax(attention_weights.view(b, q, num_heads, -1), dim=-1).view(
                b, q, num_heads, num_levels, num_points
            )

            sampling_locations = _compute_sampling_locations(
                reference_points=reference_points,
                sampling_offsets=sampling_offsets,
                num_points=num_points,
            )
            # print(f"\n[layer {layer_idx}]")
            # print("query.shape =", tuple(query.shape))
            # print("reference_points.shape =", tuple(reference_points.shape))
            # print("spatial_shapes =", spatial_shapes)
            # print("num_heads =", num_heads, "num_points =", num_points)

            # print("sampling_offsets min/max =",
            #     float(sampling_offsets.min()), float(sampling_offsets.max()))
            # print("attention_weights min/max =",
            #     float(attention_weights.min()), float(attention_weights.max()))
            # print("attention_weights sum first query/head =",
            #     float(attention_weights[0, 0, 0].sum()))

            # print("sampling_locations min/max =",
            #     float(sampling_locations.min()), float(sampling_locations.max()))

            # loc = sampling_locations[0, 0, 0, 0]  # first query, first head, first level -> [P,2]
            # print("first sampling locations =", loc[:min(8, loc.shape[0])])

            record["attention_weights"] = attention_weights.detach()
            record["sampling_locations"] = sampling_locations.detach()
            record["spatial_shapes"] = spatial_shapes.detach()
            saved_cross.append(record)

        return cross_hook

    try:
        for i, layer in enumerate(core_model.transformer.decoder.layers):
            handles.append(layer.cross_attn.register_forward_hook(make_cross_hook(i)))

        outputs = core_model(img_tensor)
        pred_logits = outputs["pred_logits"]
        pred_boxes = outputs["pred_boxes"]

    finally:
        for h in handles:
            h.remove()

    # 3) pro Query Bildkarten bauen
    q_total = pred_logits.shape[1]
    per_query_map = torch.zeros((q_total, img_h, img_w), device=device, dtype=img_tensor.dtype)

    saved_cross = sorted(saved_cross, key=lambda x: x["layer"])

    for rec in saved_cross:
        attn_w = rec["attention_weights"][0]       # [Q,H,L,P]
        samp_l = rec["sampling_locations"][0]      # [Q,H,L,P,2]
        spatial_shapes = rec["spatial_shapes"]     # [L,2]

        layer_maps = []
        for q_idx in range(q_total):
            q_map = _project_msdeform_query_to_map(
                attn_w_q=attn_w[q_idx],
                samp_locs_q=samp_l[q_idx],
                spatial_shapes=spatial_shapes,
                out_h=img_h,
                out_w=img_w,
            )
            layer_maps.append(q_map)

        layer_maps = torch.stack(layer_maps, dim=0)   # [Q,H,W]
        per_query_map += layer_maps

    per_query_map = per_query_map / max(len(saved_cross), 1)

    # 4) mit self-attn rollout über Queries aggregieren
    raw_rollout_map = (query_rollout[:, None, None] * per_query_map).sum(dim=0)
    rollout_map = normalize_cam(raw_rollout_map, mode=cam_mode)

    return {
        "rollout_map": rollout_map.detach().cpu(),
        "raw_rollout_map": raw_rollout_map.detach().cpu(),
        "query_rollout": query_rollout.detach().cpu(),
        "per_query_map": per_query_map.detach().cpu(),
        "target_query": target_query,
        "target_class": self_result["target_class"],
        "target_score": self_result["target_score"],
        "target_logit": self_result["target_logit"],
        "pred_logits": pred_logits.detach().cpu(),
        "pred_boxes": pred_boxes.detach().cpu(),
    }


def box_cxcywh_to_xyxy(boxes):
    """
    boxes: [..., 4] in cx,cy,w,h normalized
    returns: [..., 4] in x1,y1,x2,y2 normalized
    """
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)


def box_iou_xyxy(boxes1, boxes2):
    """
    boxes1: [N,4]
    boxes2: [M,4]
    returns: [N,M]
    """
    area1 = ((boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) *
             (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0))
    area2 = ((boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) *
             (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0))

    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]

    union = area1[:, None] + area2[None, :] - inter
    return inter / (union + 1e-8)


def select_target_detection(outputs, target_class=None, target_query=None, query_strategy="top_score"):
    """
    outputs:
      pred_logits: [1,Q,C]
      pred_boxes:  [1,Q,4]
    """
    pred_logits = outputs["pred_logits"]
    pred_boxes = outputs["pred_boxes"]

    logits0 = pred_logits[0]  # [Q,C]

    if target_query is None:
        if query_strategy == "top_score":
            if target_class is None:
                q = int(logits0.max(dim=-1).values.argmax().item())
                c = int(logits0[q].argmax().item())
            else:
                q = int(logits0[:, target_class].argmax().item())
                c = int(target_class)
        else:
            raise ValueError(f"Unknown query_strategy: {query_strategy}")
    else:
        q = int(target_query)
        c = int(logits0[q].argmax().item()) if target_class is None else int(target_class)

    score = logits0[q].softmax(dim=-1)[c]
    box = pred_boxes[0, q]

    return {
        "target_query": q,
        "target_class": c,
        "target_score": score.detach(),
        "target_box": box.detach(),
        "pred_logits": pred_logits.detach(),
        "pred_boxes": pred_boxes.detach(),
    }


def generate_drise_mask(image_h, image_w, grid_h=16, grid_w=16, p1=0.5, device="cpu"):
    """
    D-RISE-style random mask:
    - random binary grid
    - bilinear upsample
    - random spatial shift
    returns: [H,W]
    """
    cell_h = math.ceil(image_h / grid_h)
    cell_w = math.ceil(image_w / grid_w)

    small = (torch.rand((1, 1, grid_h, grid_w), device=device) < p1).float()

    up_h = (grid_h + 1) * cell_h
    up_w = (grid_w + 1) * cell_w

    up = F.interpolate(small, size=(up_h, up_w), mode="bilinear", align_corners=False)[0, 0]

    shift_y = torch.randint(0, cell_h, (1,), device=device).item()
    shift_x = torch.randint(0, cell_w, (1,), device=device).item()

    mask = up[shift_y:shift_y + image_h, shift_x:shift_x + image_w]
    return mask


def match_score_for_target(masked_outputs, target_class, target_box, score_power=1.0, iou_power=1.0):
    """
    Bewertet, wie gut die Ziel-Detektion im maskierten Bild noch vorhanden ist.

    masked_outputs:
      pred_logits: [1,Q,C]
      pred_boxes:  [1,Q,4]

    target_box: [4] normalized cxcywh
    """
    pred_logits = masked_outputs["pred_logits"][0]   # [Q,C]
    pred_boxes = masked_outputs["pred_boxes"][0]     # [Q,4]

    class_probs = pred_logits.softmax(dim=-1)[:, target_class]  # [Q]

    target_xyxy = box_cxcywh_to_xyxy(target_box[None])           # [1,4]
    pred_xyxy = box_cxcywh_to_xyxy(pred_boxes)                   # [Q,4]
    ious = box_iou_xyxy(target_xyxy, pred_xyxy)[0]              # [Q]

    scores = (class_probs.clamp(min=0) ** score_power) * (ious.clamp(min=0) ** iou_power)
    best_idx = int(scores.argmax().item())
    best_score = scores[best_idx]

    return {
        "best_score": best_score,
        "best_idx": best_idx,
        "best_class_prob": class_probs[best_idx].detach(),
        "best_iou": ious[best_idx].detach(),
        "all_scores": scores.detach(),
    }


def get_drise_map(
    core_model,
    img_tensor,
    target_class=None,
    target_query=None,
    query_strategy="top_score",
    num_masks=1000,
    grid_h=16,
    grid_w=16,
    p1=0.5,
    batch_size=16,
    score_power=1.0,
    iou_power=1.0,
    cam_mode="relu",
):
    device = img_tensor.device
    _, _, H, W = img_tensor.shape

    with torch.inference_mode():
        outputs = core_model(img_tensor)

    target_info = select_target_detection(
        outputs,
        target_class=target_class,
        target_query=target_query,
        query_strategy=query_strategy,
    )

    target_class = target_info["target_class"]
    target_query = target_info["target_query"]
    target_box = target_info["target_box"]

    saliency = torch.zeros((H, W), device=device, dtype=img_tensor.dtype)
    masks_sum = 0.0
    masks_buffer = []

    def process_batch(mask_batch):
        nonlocal saliency, masks_sum

        with torch.inference_mode():
            masked_imgs = img_tensor * mask_batch[:, None, :, :]
            outputs_b = core_model(masked_imgs)
            pred_logits_b = outputs_b["pred_logits"]
            pred_boxes_b = outputs_b["pred_boxes"]

            for i in range(mask_batch.shape[0]):
                masked_outputs_i = {
                    "pred_logits": pred_logits_b[i:i+1],
                    "pred_boxes": pred_boxes_b[i:i+1],
                }

                match = match_score_for_target(
                    masked_outputs_i,
                    target_class=target_class,
                    target_box=target_box,
                    score_power=score_power,
                    iou_power=iou_power,
                )

                s = match["best_score"]
                saliency += s * mask_batch[i]
                masks_sum += float(mask_batch[i].mean().item())

        del masked_imgs, outputs_b, pred_logits_b, pred_boxes_b

    for _ in range(num_masks):
        mask = generate_drise_mask(
            image_h=H,
            image_w=W,
            grid_h=grid_h,
            grid_w=grid_w,
            p1=p1,
            device=device,
        )
        masks_buffer.append(mask)

        if len(masks_buffer) == batch_size:
            batch = torch.stack(masks_buffer, dim=0)
            process_batch(batch)
            masks_buffer = []

    if len(masks_buffer) > 0:
        batch = torch.stack(masks_buffer, dim=0)
        process_batch(batch)

    raw_map = saliency / (num_masks * p1 + 1e-8)
    drise_map = normalize_cam(raw_map, mode=cam_mode)

    return {
        "drise_map": drise_map.detach().cpu(),
        "raw_drise_map": raw_map.detach().cpu(),
        "target_query": target_query,
        "target_class": target_class,
        "target_score": target_info["target_score"].cpu(),
        "target_box": target_box.cpu(),
        "pred_logits": target_info["pred_logits"].cpu(),
        "pred_boxes": target_info["pred_boxes"].cpu(),
    }

def denorm_image(img_tensor):
    """
    Erwartet [1,C,H,W] oder [C,H,W], Werte ungefähr in [0,1] oder normalisiert.
    Falls du bereits unnormalisierte Bilder hast, kannst du das vereinfachen.
    """
    if img_tensor.dim() == 4:
        img_tensor = img_tensor[0]
    img = img_tensor.detach().cpu().permute(1, 2, 0).float()

    # einfache Min-Max Darstellung
    img = img - img.min()
    img = img / (img.max() + 1e-8)
    return img.numpy()


def box_cxcywh_to_xyxy_abs(box, h, w):
    cx, cy, bw, bh = box.tolist()
    x1 = (cx - bw / 2.0) * w
    y1 = (cy - bh / 2.0) * h
    x2 = (cx + bw / 2.0) * w
    y2 = (cy + bh / 2.0) * h
    return x1, y1, x2, y2


def get_heatmap_peak(cam):
    """
    cam: [H,W] torch tensor
    returns: (x, y)
    """
    cam_flat_idx = torch.argmax(cam)
    h, w = cam.shape
    y = int(cam_flat_idx // w)
    x = int(cam_flat_idx % w)
    return x, y


def visualize_drise_result(
    img_tensor,
    drise_result,
    alpha=0.45,
    figsize=(8, 8),
    show_peak=True,
    show_box=True,
    title="D-RISE",
):
    img = denorm_image(img_tensor)
    cam = drise_result["drise_map"]
    if not isinstance(cam, torch.Tensor):
        cam = torch.tensor(cam)
    cam_np = cam.detach().cpu().numpy()

    H, W = cam.shape
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.imshow(img)
    ax.imshow(cam_np, cmap="jet", alpha=alpha)

    if show_box and "target_box" in drise_result:
        box = drise_result["target_box"]
        if not isinstance(box, torch.Tensor):
            box = torch.tensor(box)
        x1, y1, x2, y2 = box_cxcywh_to_xyxy_abs(box.cpu(), H, W)
        rect = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=2,
            edgecolor="lime",
            facecolor="none",
        )
        ax.add_patch(rect)

    if show_peak:
        x_peak, y_peak = get_heatmap_peak(cam.cpu())
        ax.scatter([x_peak], [y_peak], s=60, marker="x")

    tq = drise_result.get("target_query", None)
    tc = drise_result.get("target_class", None)
    score = drise_result.get("target_score", None)

    if score is not None and isinstance(score, torch.Tensor):
        score = float(score.item())

    title_text = title
    if tq is not None and tc is not None:
        if score is not None:
            title_text += f" | q={tq}, c={tc}, score={score:.3f}"
        else:
            title_text += f" | q={tq}, c={tc}"

    ax.set_title(title_text)
    ax.axis("off")
    plt.tight_layout()
    plt.show()


def normalize_signed_cam(cam, eps=1e-8):
    cam = cam.detach()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + eps)
    return cam


def is_point_in_box(point_xy, box_cxcywh, image_size):
    x, y = point_xy
    H, W = image_size

    cx, cy, bw, bh = box_cxcywh

    x1 = (cx - bw / 2.0) * W
    y1 = (cy - bh / 2.0) * H
    x2 = (cx + bw / 2.0) * W
    y2 = (cy + bh / 2.0) * H

    return (x1 <= x <= x2) and (y1 <= y <= y2)


def get_heatmap_max_point(heatmap, require_positive=True):
    heatmap = np.asarray(heatmap)

    if heatmap.ndim != 2:
        raise ValueError(f"heatmap must be 2D, got {heatmap.shape}")

    if heatmap.size == 0:
        return None, None

    if require_positive and np.max(heatmap) <= 0:
        return None, None

    flat_idx = np.argmax(heatmap)
    y, x = np.unravel_index(flat_idx, heatmap.shape)
    max_val = float(heatmap[y, x])

    return (int(x), int(y)), max_val


def get_pointing_game_result(heatmap, gt_boxes_rel, require_positive=True):
    if gt_boxes_rel is None or len(gt_boxes_rel) == 0:
        return {
            "valid": False,
            "hit": False,
            "max_point": None,
            "max_val": None,
            "hit_box_idx": -1,
            "reason": "no_gt_boxes",
        }

    max_point, max_val = get_heatmap_max_point(
        heatmap=heatmap,
        require_positive=require_positive
    )

    if max_point is None:
        return {
            "valid": False,
            "hit": False,
            "max_point": None,
            "max_val": None,
            "hit_box_idx": -1,
            "reason": "no_valid_heatmap_peak",
        }

    H, W = heatmap.shape

    for idx, box in enumerate(gt_boxes_rel):
        if is_point_in_box(max_point, box, image_size=(H, W)):
            return {
                "valid": True,
                "hit": True,
                "max_point": max_point,
                "max_val": max_val,
                "hit_box_idx": idx,
                "reason": "hit",
            }

    return {
        "valid": True,
        "hit": False,
        "max_point": max_point,
        "max_val": max_val,
        "hit_box_idx": -1,
        "reason": "miss",
    }


def rel_box_cxcywh_to_xyxy_abs(box_cxcywh, image_size):
    """
    box_cxcywh: (cx, cy, w, h) in relative coords [0,1]
    image_size: (H, W)

    returns:
        x1, y1, x2, y2 in pixel coords (float)
    """
    H, W = image_size
    cx, cy, bw, bh = box_cxcywh

    x1 = (cx - bw / 2.0) * W
    y1 = (cy - bh / 2.0) * H
    x2 = (cx + bw / 2.0) * W
    y2 = (cy + bh / 2.0) * H

    return x1, y1, x2, y2


def box_to_mask(box_cxcywh, image_size):
    """
    Erzeugt binäre Maske [H,W] für eine relative cxcywh-Box.
    """
    H, W = image_size
    x1, y1, x2, y2 = rel_box_cxcywh_to_xyxy_abs(box_cxcywh, image_size)

    x1 = int(np.floor(np.clip(x1, 0, W)))
    y1 = int(np.floor(np.clip(y1, 0, H)))
    x2 = int(np.ceil(np.clip(x2, 0, W)))
    y2 = int(np.ceil(np.clip(y2, 0, H)))

    mask = np.zeros((H, W), dtype=bool)
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = True
    return mask


def boxes_to_union_mask(gt_boxes_rel, image_size):
    """
    Union aller GT-Boxen als binäre Maske [H,W].
    """
    H, W = image_size
    union_mask = np.zeros((H, W), dtype=bool)

    if gt_boxes_rel is None:
        return union_mask

    for box in gt_boxes_rel:
        union_mask |= box_to_mask(box, image_size)

    return union_mask


def get_mass_in_bounding_box(heatmap, gt_boxes_rel, use_absolute_values=False, clamp_negative_to_zero=True):
    """
    Misst, wie viel Heatmap-Masse innerhalb der Union aller GT-Boxen liegt.

    Args:
        heatmap: 2D array [H,W]
        gt_boxes_rel: Liste relativer cxcywh-Boxen
        use_absolute_values:
            Falls True, nutze abs(heatmap)
        clamp_negative_to_zero:
            Falls True, setze negative Werte auf 0
            (typisch für Saliency-Heatmaps sinnvoll)

    Returns:
        dict mit mass_in_box, total_mass, ratio, ...
    """
    heatmap = np.asarray(heatmap)

    if heatmap.ndim != 2:
        raise ValueError(f"heatmap must be 2D, got {heatmap.shape}")

    if gt_boxes_rel is None or len(gt_boxes_rel) == 0:
        return {
            "valid": False,
            "mass_in_box": None,
            "total_mass": None,
            "mass_ratio": None,
            "reason": "no_gt_boxes",
        }

    hm = heatmap.astype(np.float32)

    if use_absolute_values:
        hm = np.abs(hm)
    elif clamp_negative_to_zero:
        hm = np.maximum(hm, 0.0)

    total_mass = float(hm.sum())
    if total_mass <= 0:
        return {
            "valid": False,
            "mass_in_box": 0.0,
            "total_mass": 0.0,
            "mass_ratio": None,
            "reason": "no_positive_mass",
        }

    H, W = hm.shape
    union_mask = boxes_to_union_mask(gt_boxes_rel, image_size=(H, W))
    mass_in_box = float(hm[union_mask].sum())
    mass_ratio = mass_in_box / (total_mass + 1e-8)

    return {
        "valid": True,
        "mass_in_box": mass_in_box,
        "total_mass": total_mass,
        "mass_ratio": mass_ratio,
        "reason": "ok",
    }


def threshold_heatmap(heatmap, threshold=0.5, normalize_first=True):
    """
    Thresholdet Heatmap zu binärer Maske.

    Args:
        heatmap: 2D array
        threshold:
            Wenn normalize_first=True, typischerweise in [0,1]
        normalize_first:
            Min-Max-Normalisierung vor Thresholding

    Returns:
        binary_mask: bool [H,W]
        norm_heatmap: float [H,W]
    """
    heatmap = np.asarray(heatmap)

    if heatmap.ndim != 2:
        raise ValueError(f"heatmap must be 2D, got {heatmap.shape}")

    hm = heatmap.astype(np.float32)

    if normalize_first:
        hm_min = hm.min()
        hm_max = hm.max()
        hm = (hm - hm_min) / (hm_max - hm_min + 1e-8)

    binary_mask = hm >= threshold
    return binary_mask, hm


def compute_iou(mask_a, mask_b):
    """
    IoU zwischen zwei bool-Masken [H,W].
    """
    mask_a = np.asarray(mask_a).astype(bool)
    mask_b = np.asarray(mask_b).astype(bool)

    if mask_a.shape != mask_b.shape:
        raise ValueError(f"Shape mismatch: {mask_a.shape} vs {mask_b.shape}")

    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()

    if union == 0:
        return 0.0

    return float(inter / union)


def get_iou_after_thresholding(
    heatmap,
    gt_boxes_rel,
    threshold=0.5,
    normalize_first=True,
):
    """
    Thresholdet die Heatmap und vergleicht sie per IoU mit der Union der GT-Boxen.

    Returns:
        dict mit iou, pred_mask, gt_mask, ...
    """
    heatmap = np.asarray(heatmap)

    if heatmap.ndim != 2:
        raise ValueError(f"heatmap must be 2D, got {heatmap.shape}")

    if gt_boxes_rel is None or len(gt_boxes_rel) == 0:
        return {
            "valid": False,
            "iou": None,
            "pred_area": None,
            "gt_area": None,
            "intersection": None,
            "union": None,
            "reason": "no_gt_boxes",
        }

    H, W = heatmap.shape
    pred_mask, norm_heatmap = threshold_heatmap(
        heatmap,
        threshold=threshold,
        normalize_first=normalize_first,
    )
    gt_mask = boxes_to_union_mask(gt_boxes_rel, image_size=(H, W))

    inter = int(np.logical_and(pred_mask, gt_mask).sum())
    union = int(np.logical_or(pred_mask, gt_mask).sum())
    iou = float(inter / union) if union > 0 else 0.0

    return {
        "valid": True,
        "iou": iou,
        "pred_area": int(pred_mask.sum()),
        "gt_area": int(gt_mask.sum()),
        "intersection": inter,
        "union": union,
        "threshold": threshold,
        "reason": "ok",
    }


def summarize_mass_in_box(results):
    valid_results = [r for r in results if r["valid"] and r["mass_ratio"] is not None]

    return {
        "num_total": len(results),
        "num_valid": len(valid_results),
        "mean_mass_ratio": float(np.mean([r["mass_ratio"] for r in valid_results])) if valid_results else None,
        "median_mass_ratio": float(np.median([r["mass_ratio"] for r in valid_results])) if valid_results else None,
    }


def summarize_iou_after_thresholding(results):
    valid_results = [r for r in results if r["valid"] and r["iou"] is not None]

    return {
        "num_total": len(results),
        "num_valid": len(valid_results),
        "mean_iou": float(np.mean([r["iou"] for r in valid_results])) if valid_results else None,
        "median_iou": float(np.median([r["iou"] for r in valid_results])) if valid_results else None,
    }


def summarize_pointing_game(results):
    valid_results = [r for r in results if r["valid"]]
    num_valid = len(valid_results)
    num_hits = sum(int(r["hit"]) for r in valid_results)

    return {
        "num_total": len(results),
        "num_valid": num_valid,
        "num_hits": num_hits,
        "num_misses": num_valid - num_hits,
        "pointing_game_score": num_hits / num_valid if num_valid > 0 else None,
    }

def build_xai_per_sample_rows(
    all_pg_gradcam,
    all_pg_rollout,
    all_pg_drise,
    all_mbb_gradcam,
    all_mbb_rollout,
    all_mbb_drise,
    all_iou_gradcam,
    all_iou_rollout,
    all_iou_drise,
):
    """
    Baut eine flache per-sample Tabelle:
    sample_id,image_id,file_name,method,pg,ebpg,iou,...

    Wichtig:
    - pg kommt aus pointing_game['hit']
    - ebpg kommt aus mass_in_bounding_box['mass_ratio']
    - iou kommt aus iou['iou']
    """

    method_data = {
        "gradcam": {
            "pg": all_pg_gradcam,
            "ebpg": all_mbb_gradcam,
            "iou": all_iou_gradcam,
        },
        "rollout": {
            "pg": all_pg_rollout,
            "ebpg": all_mbb_rollout,
            "iou": all_iou_rollout,
        },
        "drise": {
            "pg": all_pg_drise,
            "ebpg": all_mbb_drise,
            "iou": all_iou_drise,
        },
    }

    rows = []

    for method, data in method_data.items():
        pg_by_image = {r["image_id"]: r for r in data["pg"]}
        ebpg_by_image = {r["image_id"]: r for r in data["ebpg"]}
        iou_by_image = {r["image_id"]: r for r in data["iou"]}

        common_image_ids = sorted(
            set(pg_by_image.keys())
            & set(ebpg_by_image.keys())
            & set(iou_by_image.keys())
        )

        for image_id in common_image_ids:
            pg_r = pg_by_image[image_id]
            ebpg_r = ebpg_by_image[image_id]
            iou_r = iou_by_image[image_id]

            # Nur Samples verwenden, bei denen alle drei Metriken gültig sind.
            # Falls du ungültige behalten willst, sag Bescheid.
            if not pg_r.get("valid", False):
                continue
            if not ebpg_r.get("valid", False):
                continue
            if not iou_r.get("valid", False):
                continue

            rows.append({
                "sample_id": str(image_id),
                "image_id": int(image_id),
                "file_name": pg_r.get("file_name"),
                "method": method,

                # Für McNemar: binär 0/1
                "pg": int(bool(pg_r.get("hit", False))),

                # Für Wilcoxon: kontinuierliche Werte
                "ebpg": float(ebpg_r["mass_ratio"]),
                "iou": float(iou_r["iou"]),

                # Zusatzinfos, praktisch fürs Debugging
                "num_gt_boxes": int(pg_r.get("num_gt_boxes", 0)),
                "target_query": int(pg_r.get("target_query", -1)),
                "target_class": int(pg_r.get("target_class", -1)),
                "target_score": float(pg_r.get("target_score", 0.0)),
            })

    return rows


def save_xai_per_sample_csv(rows, output_json_path):
    import csv

    output_json_path = str(output_json_path)
    csv_path = output_json_path.replace(".json", "_per_sample_metrics.csv")

    fieldnames = [
        "sample_id",
        "image_id",
        "file_name",
        "method",
        "pg",
        "ebpg",
        "iou",
        "num_gt_boxes",
        "target_query",
        "target_class",
        "target_score",
    ]

    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return csv_path
def main(
    model_path="checkpoint_best_total.pth",
    test_root="C:/Users/chris/Documents/GitHub/DINO-LungDet/data/processed",
    coco_path="C:/Users/chris/Documents/GitHub/DINO-LungDet/data/processed/test/_annotations.coco.json",
    output_path="outputs/results_eval_1.json",
    device=None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    model = RFDETRMedium(
        pretrain_weights=model_path,
        device=device,
    )

    with open(coco_path, "r", encoding="utf-8") as f:
        coco_data = json.load(f)

    images_by_id = {img["id"]: img for img in coco_data["images"]}
    anns_by_image_id = defaultdict(list)
    for ann in coco_data["annotations"]:
        anns_by_image_id[ann["image_id"]].append(ann)

    core_model = model.model.model.to(device)
    core_model.eval()

    all_pg_gradcam = []
    all_pg_rollout = []
    all_pg_drise = []

    all_mbb_gradcam = []
    all_mbb_rollout = []
    all_mbb_drise = []

    all_iou_gradcam = []
    all_iou_rollout = []
    all_iou_drise = []

    for idx, (image_id, img_info) in enumerate(images_by_id.items()):
        print(f"Processing image {idx + 1}/{len(images_by_id)}: ID={image_id}")

        img_path = os.path.join(test_root, img_info["file_name"])
        image, img_tensor = load_image_as_tensor(img_path, transform, device)

        gt_boxes_rel, gt_anns = get_gt_boxes_for_image(
            image_id=image_id,
            images_by_id=images_by_id,
            anns_by_image_id=anns_by_image_id,
            category_id=None,
        )

        print("  Running Grad-CAM...")
        gradcam_result = get_conv_gradcam(
            core_model=core_model,
            img_tensor=img_tensor,
            pil_img=image,
            target_layer=core_model.backbone[0].projector.stages[0][0].cv2.conv,
            target_class=0,
            target_query=None,
            query_strategy="top_score",
            cam_mode="negative",
            use_box_term=False,
            box_weight=0.5,
            show=False,
            draw_box=True,
        )

        cam = gradcam_result["cam"]
        cam_norm = normalize_signed_cam(cam)

        print("        Pointing Game")
        pg_result_gradcam = get_pointing_game_result(
            heatmap=cam_norm.numpy(),
            gt_boxes_rel=gt_boxes_rel,
            require_positive=True,
        )
        pg_result_gradcam.update({
            "method": "gradcam",
            "image_id": image_id,
            "file_name": img_info["file_name"],
            "num_gt_boxes": len(gt_boxes_rel),
            "target_query": gradcam_result["target_query"],
            "target_class": gradcam_result["target_class"],
            "target_score": float(gradcam_result["target_score"]),
        })
        all_pg_gradcam.append(pg_result_gradcam)

        print("        Mass in Bounding Box")
        mbb_result_gradcam = get_mass_in_bounding_box(
            heatmap=cam_norm.numpy(),
            gt_boxes_rel=gt_boxes_rel,
        )
        mbb_result_gradcam.update({
            "method": "gradcam",
            "image_id": image_id,
            "file_name": img_info["file_name"],
            "num_gt_boxes": len(gt_boxes_rel),
            "target_query": gradcam_result["target_query"],
            "target_class": gradcam_result["target_class"],
            "target_score": float(gradcam_result["target_score"]),
        })
        all_mbb_gradcam.append(mbb_result_gradcam)

        print("        IoU")
        iou_result_gradcam = get_iou_after_thresholding(
            heatmap=cam_norm.numpy(),
            gt_boxes_rel=gt_boxes_rel,
            threshold=0.5,
        )
        iou_result_gradcam.update({
            "method": "gradcam",
            "image_id": image_id,
            "file_name": img_info["file_name"],
            "num_gt_boxes": len(gt_boxes_rel),
            "target_query": gradcam_result["target_query"],
            "target_class": gradcam_result["target_class"],
            "target_score": float(gradcam_result["target_score"]),
        })
        all_iou_gradcam.append(iou_result_gradcam)

        print("  Running Attention Rollout...")
        rollout_result = get_decoder_self_cross_rollout_map(
            core_model=core_model,
            img_tensor=img_tensor,
            target_class=0,
            target_query=None,
            query_strategy="top_score",
            cam_mode="minmax",
            add_self_attn_residual=True,
        )

        target_query = rollout_result["target_query"]
        direct_map = rollout_result["per_query_map"][target_query]
        direct_map_norm = direct_map - direct_map.min()
        direct_map_norm = direct_map_norm / (direct_map_norm.max() + 1e-8)

        print("        Pointing Game")
        pg_result_rollout = get_pointing_game_result(
            heatmap=direct_map_norm.numpy(),
            gt_boxes_rel=gt_boxes_rel,
            require_positive=True,
        )
        pg_result_rollout.update({
            "method": "rollout",
            "image_id": image_id,
            "file_name": img_info["file_name"],
            "num_gt_boxes": len(gt_boxes_rel),
            "target_query": rollout_result["target_query"],
            "target_class": rollout_result["target_class"],
            "target_score": float(rollout_result["target_score"]),
        })
        all_pg_rollout.append(pg_result_rollout)

        print("        Mass in Bounding Box")
        mbb_result_rollout = get_mass_in_bounding_box(
            heatmap=direct_map_norm.numpy(),
            gt_boxes_rel=gt_boxes_rel,
        )
        mbb_result_rollout.update({
            "method": "rollout",
            "image_id": image_id,
            "file_name": img_info["file_name"],
            "num_gt_boxes": len(gt_boxes_rel),
            "target_query": rollout_result["target_query"],
            "target_class": rollout_result["target_class"],
            "target_score": float(rollout_result["target_score"]),
        })
        all_mbb_rollout.append(mbb_result_rollout)

        print("        IoU")
        iou_result_rollout = get_iou_after_thresholding(
            heatmap=direct_map_norm.numpy(),
            gt_boxes_rel=gt_boxes_rel,
            threshold=0.5,
        )
        iou_result_rollout.update({
            "method": "rollout",
            "image_id": image_id,
            "file_name": img_info["file_name"],
            "num_gt_boxes": len(gt_boxes_rel),
            "target_query": rollout_result["target_query"],
            "target_class": rollout_result["target_class"],
            "target_score": float(rollout_result["target_score"]),
        })
        all_iou_rollout.append(iou_result_rollout)

        print("  Running D-RISE...")
        drise_result = get_drise_map(
            core_model=core_model,
            img_tensor=img_tensor,
            num_masks=1028,
            grid_h=16,
            grid_w=16,
            p1=0.5,
            batch_size=16,
            iou_power=2.0,
        )

        heatmap = drise_result["drise_map"].numpy()

        print("        Pointing Game")
        pg_result_drise = get_pointing_game_result(
            heatmap=heatmap,
            gt_boxes_rel=gt_boxes_rel,
            require_positive=True,
        )
        pg_result_drise.update({
            "method": "drise",
            "image_id": image_id,
            "file_name": img_info["file_name"],
            "num_gt_boxes": len(gt_boxes_rel),
            "target_query": drise_result["target_query"],
            "target_class": drise_result["target_class"],
            "target_score": float(drise_result["target_score"]),
        })
        all_pg_drise.append(pg_result_drise)

        print("        Mass in Bounding Box")
        mbb_result_drise = get_mass_in_bounding_box(
            heatmap=heatmap,
            gt_boxes_rel=gt_boxes_rel,
        )
        mbb_result_drise.update({
            "method": "drise",
            "image_id": image_id,
            "file_name": img_info["file_name"],
            "num_gt_boxes": len(gt_boxes_rel),
            "target_query": drise_result["target_query"],
            "target_class": drise_result["target_class"],
            "target_score": float(drise_result["target_score"]),
        })
        all_mbb_drise.append(mbb_result_drise)

        print("        IoU")
        iou_result_drise = get_iou_after_thresholding(
            heatmap=heatmap,
            gt_boxes_rel=gt_boxes_rel,
            threshold=0.5,
        )
        iou_result_drise.update({
            "method": "drise",
            "image_id": image_id,
            "file_name": img_info["file_name"],
            "num_gt_boxes": len(gt_boxes_rel),
            "target_query": drise_result["target_query"],
            "target_class": drise_result["target_class"],
            "target_score": float(drise_result["target_score"]),
        })
        all_iou_drise.append(iou_result_drise)

    payload = {
        "summary": {
            "gradcam": summarize_pointing_game(all_pg_gradcam),
            "rollout": summarize_pointing_game(all_pg_rollout),
            "drise": summarize_pointing_game(all_pg_drise),
            "mass_in_box_gradcam": summarize_mass_in_box(all_mbb_gradcam),
            "mass_in_box_rollout": summarize_mass_in_box(all_mbb_rollout),
            "mass_in_box_drise": summarize_mass_in_box(all_mbb_drise),
            "iou_gradcam": summarize_iou_after_thresholding(all_iou_gradcam),
            "iou_rollout": summarize_iou_after_thresholding(all_iou_rollout),
            "iou_drise": summarize_iou_after_thresholding(all_iou_drise),
        },
        "pointing_game": {
            "gradcam": all_pg_gradcam,
            "rollout": all_pg_rollout,
            "drise": all_pg_drise,
        },
        "mass_in_bounding_box": {
            "gradcam": all_mbb_gradcam,
            "rollout": all_mbb_rollout,
            "drise": all_mbb_drise,
        },
        "iou": {
            "gradcam": all_iou_gradcam,
            "rollout": all_iou_rollout,
            "drise": all_iou_drise,
        },
    }
    xai_per_sample_rows = build_xai_per_sample_rows(
        all_pg_gradcam=all_pg_gradcam,
        all_pg_rollout=all_pg_rollout,
        all_pg_drise=all_pg_drise,
        all_mbb_gradcam=all_mbb_gradcam,
        all_mbb_rollout=all_mbb_rollout,
        all_mbb_drise=all_mbb_drise,
        all_iou_gradcam=all_iou_gradcam,
        all_iou_rollout=all_iou_rollout,
        all_iou_drise=all_iou_drise,
    )

    payload["per_sample_metrics"] = xai_per_sample_rows

    per_sample_csv_path = save_xai_per_sample_csv(
        rows=xai_per_sample_rows,
        output_json_path=output_path,
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"\nSaved results to: {output_path}")
    print("Grad-CAM:", payload["summary"]["gradcam"])
    print("Rollout:", payload["summary"]["rollout"])
    print("D-RISE:", payload["summary"]["drise"])
    print(f"Saved per-sample XAI metrics CSV to: {per_sample_csv_path}")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run XAI evaluation for RF-DETR.")
    parser.add_argument("--model-path", default="checkpoint_best_ema.pth")
    parser.add_argument("--test-root", default="data/processed/test")
    parser.add_argument("--coco-path", default="data/processed/test/_annotations.coco.json")
    parser.add_argument("--output-path", default="outputs/results_eval.json")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    main(
        model_path=args.model_path,
        test_root=args.test_root,
        coco_path=args.coco_path,
        output_path=args.output_path,
        device=args.device,
    )
