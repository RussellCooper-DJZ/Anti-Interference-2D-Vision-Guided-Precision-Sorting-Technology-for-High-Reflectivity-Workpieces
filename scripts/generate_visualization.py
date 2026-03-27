"""
generate_visualization.py — Synthetic Dataset Complete Visualization Generator
:Author: RussellCooper

Generates:
  1. 8 scenes × 7 lights = 56 sample images (image + mask + edge + overlay)
  2. Scene comparison charts (each scene vs 7 lights)
  3. Light comparison charts (each light vs 8 scenes)
  4. Statistical distribution pie charts + bar charts
  5. Pixel brightness histograms (per scene)
  6. Edge density heatmaps (per scene)
  7. Highlight area ratio heatmap (scene × light matrix)
  8. Mask coverage bar chart
  9. Full 8×7 overview grid
  10. Deep 4-view per scene
"""

import os
import sys
import json
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.synth_national_scenes import NationalSceneGenerator, SceneType, LightType

# ============================================================
# Configuration
# ============================================================
OUT_DIR = Path("/home/ubuntu/repo/docs/visualization")
SEED    = 42
IMG_H   = 256
IMG_W   = 256

SCENES = list(SceneType)
LIGHTS = list(LightType)

SCENE_LABELS = {
    SceneType.SHIPYARD:     "Shipyard\n(Ship Hull)",
    SceneType.STEEL_MILL:   "Steel Mill\n(Hot Coil)",
    SceneType.BRIDGE:       "Bridge\n(Steel Box Girder)",
    SceneType.PORT_CRANE:   "Port Crane\n(Container Top)",
    SceneType.RAILWAY:      "Railway\n(Al Body Skin)",
    SceneType.CURTAIN_WALL: "Curtain Wall\n(Al Frame)",
    SceneType.PIPELINE:     "Pipeline\n(LNG Tank)",
    SceneType.WIND_TURBINE: "Wind Turbine\n(Tower/Nacelle)",
}

LIGHT_LABELS = {
    LightType.SIDE_SUN:     "Side Sun",
    LightType.TOP_SKYLIGHT: "Top Skylight",
    LightType.WATER_REFL:   "Water Refl.",
    LightType.OVERCAST:     "Overcast",
    LightType.NIGHT_LED:    "Night LED",
    LightType.WELD_ARC:     "Weld Arc",
    LightType.MIXED:        "Mixed",
}

SCENE_COLORS = [
    '#2E86AB', '#A23B72', '#F18F01', '#C73E1D',
    '#3B8F2B', '#44BBA4', '#E94F37', '#8A6F3E'
]
LIGHT_COLORS = [
    '#FFD166', '#06D6A0', '#118AB2', '#9BB7D4',
    '#EF476F', '#26547C', '#FF9F43'
]

BG_DARK  = '#0d1117'
BG_MID   = '#161b22'
BG_PANEL = '#1a1a2e'


def setup_dirs():
    for sub in ['samples', 'scene_compare', 'light_compare',
                'stats', 'brightness', 'edge_density', 'overview']:
        (OUT_DIR / sub).mkdir(parents=True, exist_ok=True)


def bgr2rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ============================================================
# Phase 1: Generate 56 samples
# ============================================================
def generate_all_samples(gen):
    print("\n[Phase 1] Generating 56 scene x light samples...")
    samples = {}
    total = len(SCENES) * len(LIGHTS)
    idx = 0
    for scene in SCENES:
        samples[scene] = {}
        for light in LIGHTS:
            idx += 1
            s = gen.generate(scene=scene.value, light=light.value)
            samples[scene][light] = s
            vis = NationalSceneGenerator._make_preview(s)
            fname = OUT_DIR / 'samples' / f"{scene.value}_{light.value}.png"
            cv2.imwrite(str(fname), vis)
            print(f"  [{idx:2d}/{total}] {scene.value:15s} + {light.value:12s} -> OK")
    return samples


# ============================================================
# Phase 2A: Scene comparison (1 scene, 7 lights)
# ============================================================
def make_scene_compare(samples):
    print("\n[Phase 2A] Scene comparison charts...")
    for scene in SCENES:
        fig, axes = plt.subplots(1, len(LIGHTS), figsize=(len(LIGHTS) * 2.4, 3.2))
        fig.patch.set_facecolor(BG_DARK)
        fig.suptitle(
            f"Scene: {SCENE_LABELS[scene].replace(chr(10),' ')}  —  7 Light Conditions",
            color='white', fontsize=11, fontweight='bold', y=1.02
        )
        for j, light in enumerate(LIGHTS):
            img = bgr2rgb(samples[scene][light]['image'])
            axes[j].imshow(img)
            axes[j].set_title(LIGHT_LABELS[light], color='#aaddff', fontsize=8, pad=3)
            axes[j].axis('off')
        plt.tight_layout(pad=0.3)
        fname = OUT_DIR / 'scene_compare' / f"{scene.value}_all_lights.png"
        plt.savefig(str(fname), dpi=130, bbox_inches='tight', facecolor=BG_DARK)
        plt.close()
        print(f"  {fname.name}")


# ============================================================
# Phase 2B: Light comparison (1 light, 8 scenes)
# ============================================================
def make_light_compare(samples):
    print("\n[Phase 2B] Light comparison charts...")
    for light in LIGHTS:
        fig, axes = plt.subplots(1, len(SCENES), figsize=(len(SCENES) * 2.4, 3.2))
        fig.patch.set_facecolor(BG_DARK)
        fig.suptitle(
            f"Light: {LIGHT_LABELS[light]}  —  8 Scene Types",
            color='white', fontsize=11, fontweight='bold', y=1.02
        )
        for j, scene in enumerate(SCENES):
            img = bgr2rgb(samples[scene][light]['image'])
            axes[j].imshow(img)
            axes[j].set_title(SCENE_LABELS[scene], color='#ffddaa', fontsize=7, pad=3)
            axes[j].axis('off')
        plt.tight_layout(pad=0.3)
        fname = OUT_DIR / 'light_compare' / f"{light.value}_all_scenes.png"
        plt.savefig(str(fname), dpi=130, bbox_inches='tight', facecolor=BG_DARK)
        plt.close()
        print(f"  {fname.name}")


# ============================================================
# Phase 2C: Statistical distribution
# ============================================================
def make_stats(samples):
    print("\n[Phase 2C] Statistical distribution charts...")

    # Scene weight pie + bar
    svals   = [NationalSceneGenerator.SCENE_WEIGHTS[s] for s in SCENES]
    slabels = [SCENE_LABELS[s].replace('\n', '\n') for s in SCENES]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.patch.set_facecolor(BG_DARK)

    wedges, texts, autotexts = axes[0].pie(
        svals, labels=slabels, colors=SCENE_COLORS,
        autopct='%1.1f%%', startangle=140,
        textprops={'color': 'white', 'fontsize': 8},
        wedgeprops={'edgecolor': BG_DARK, 'linewidth': 1.5}
    )
    for at in autotexts:
        at.set_fontsize(8); at.set_color('#ffff99')
    axes[0].set_title('Scene Sampling Weight Distribution',
                      color='white', fontsize=12, pad=10)
    axes[0].set_facecolor(BG_DARK)

    bars = axes[1].barh(
        [SCENE_LABELS[s].replace('\n', ' ') for s in SCENES],
        [int(v * 1000) for v in svals],
        color=SCENE_COLORS, edgecolor='#333355', linewidth=0.8
    )
    axes[1].set_xlabel('Samples per 1000', color='white', fontsize=10)
    axes[1].set_title('Scene Frequency (per 1000 Samples)', color='white', fontsize=12)
    axes[1].tick_params(colors='white', labelsize=8)
    axes[1].set_facecolor(BG_MID)
    for sp in ['top', 'right']:
        axes[1].spines[sp].set_visible(False)
    for sp in ['bottom', 'left']:
        axes[1].spines[sp].set_color('#555577')
    for bar, val in zip(bars, svals):
        axes[1].text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                     f'{int(val*1000)}', va='center', color='#aaffcc', fontsize=8)

    plt.tight_layout(pad=1.5)
    plt.savefig(str(OUT_DIR / 'stats' / 'scene_distribution.png'),
                dpi=130, bbox_inches='tight', facecolor=BG_DARK)
    plt.close()
    print("  scene_distribution.png")

    # Light weight pie + bar
    lvals   = [NationalSceneGenerator.LIGHT_WEIGHTS[l] for l in LIGHTS]
    llabels = [LIGHT_LABELS[l] for l in LIGHTS]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(BG_DARK)

    wedges, texts, autotexts = axes[0].pie(
        lvals, labels=llabels, colors=LIGHT_COLORS,
        autopct='%1.1f%%', startangle=100,
        textprops={'color': 'white', 'fontsize': 9},
        wedgeprops={'edgecolor': BG_DARK, 'linewidth': 1.5}
    )
    for at in autotexts:
        at.set_fontsize(8); at.set_color('#ffff99')
    axes[0].set_title('Light Type Weight Distribution', color='white', fontsize=12, pad=10)
    axes[0].set_facecolor(BG_DARK)

    bars = axes[1].barh(llabels, [int(v * 1000) for v in lvals],
                        color=LIGHT_COLORS, edgecolor='#333355', linewidth=0.8)
    axes[1].set_xlabel('Samples per 1000', color='white', fontsize=10)
    axes[1].set_title('Light Frequency (per 1000 Samples)', color='white', fontsize=12)
    axes[1].tick_params(colors='white', labelsize=8)
    axes[1].set_facecolor(BG_MID)
    for sp in ['top', 'right']:
        axes[1].spines[sp].set_visible(False)
    for sp in ['bottom', 'left']:
        axes[1].spines[sp].set_color('#555577')
    for bar, val in zip(bars, lvals):
        axes[1].text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                     f'{int(val*1000)}', va='center', color='#aaffcc', fontsize=8)

    plt.tight_layout(pad=1.5)
    plt.savefig(str(OUT_DIR / 'stats' / 'light_distribution.png'),
                dpi=130, bbox_inches='tight', facecolor=BG_DARK)
    plt.close()
    print("  light_distribution.png")


# ============================================================
# Phase 2D: Brightness histograms
# ============================================================
def make_brightness_histograms(samples):
    print("\n[Phase 2D] Brightness histograms...")
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    fig.patch.set_facecolor(BG_DARK)
    fig.suptitle(
        'Pixel Brightness Distribution per Scene (7 Light Types Overlaid)',
        color='white', fontsize=13, fontweight='bold', y=1.01
    )
    axes = axes.flatten()
    for i, scene in enumerate(SCENES):
        ax = axes[i]
        ax.set_facecolor(BG_MID)
        for j, light in enumerate(LIGHTS):
            gray = cv2.cvtColor(samples[scene][light]['image'],
                                cv2.COLOR_BGR2GRAY).flatten()
            ax.hist(gray, bins=64, range=(0, 255),
                    color=LIGHT_COLORS[j], alpha=0.55, linewidth=0,
                    label=LIGHT_LABELS[light])
        ax.set_title(SCENE_LABELS[scene].replace('\n', ' '),
                     color='white', fontsize=9, pad=4)
        ax.set_xlim(0, 255)
        ax.set_xlabel('Brightness', color='#aaaaaa', fontsize=7)
        ax.set_ylabel('Pixel Count', color='#aaaaaa', fontsize=7)
        ax.tick_params(colors='#888888', labelsize=7)
        for sp in ['top', 'right']:
            ax.spines[sp].set_visible(False)
        for sp in ['bottom', 'left']:
            ax.spines[sp].set_color('#333344')
        if i == 0:
            ax.legend(fontsize=6.5, labelcolor='white',
                      facecolor=BG_DARK, edgecolor='#333344', loc='upper left')
    plt.tight_layout(pad=1.2)
    plt.savefig(str(OUT_DIR / 'brightness' / 'brightness_histograms.png'),
                dpi=130, bbox_inches='tight', facecolor=BG_DARK)
    plt.close()
    print("  brightness_histograms.png")


# ============================================================
# Phase 2E: Edge density heatmaps
# ============================================================
def make_edge_density(samples):
    print("\n[Phase 2E] Edge density heatmaps...")
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    fig.patch.set_facecolor(BG_DARK)
    fig.suptitle(
        'Edge Density Heatmap per Scene (Average over 7 Light Types)',
        color='white', fontsize=13, fontweight='bold', y=1.01
    )
    axes = axes.flatten()
    for i, scene in enumerate(SCENES):
        ax = axes[i]
        acc = np.zeros((IMG_H, IMG_W), dtype=np.float32)
        for light in LIGHTS:
            edge = cv2.resize(samples[scene][light]['edge'], (IMG_W, IMG_H))
            acc += edge.astype(np.float32)
        acc /= len(LIGHTS)
        acc_smooth = cv2.GaussianBlur(acc, (15, 15), 0)
        im = ax.imshow(acc_smooth, cmap='plasma', vmin=0, vmax=acc_smooth.max())
        ax.set_title(SCENE_LABELS[scene].replace('\n', ' '),
                     color='white', fontsize=9, pad=4)
        ax.axis('off')
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(colors='#aaaaaa', labelsize=6)
    plt.tight_layout(pad=1.2)
    plt.savefig(str(OUT_DIR / 'edge_density' / 'edge_density_heatmaps.png'),
                dpi=130, bbox_inches='tight', facecolor=BG_DARK)
    plt.close()
    print("  edge_density_heatmaps.png")


# ============================================================
# Phase 2F: Highlight ratio heatmap (scene × light)
# ============================================================
def make_highlight_ratio(samples):
    print("\n[Phase 2F] Highlight area ratio heatmap...")
    THRESHOLD = 220
    ratio_matrix = np.zeros((len(SCENES), len(LIGHTS)), dtype=np.float32)
    for i, scene in enumerate(SCENES):
        for j, light in enumerate(LIGHTS):
            gray = cv2.cvtColor(samples[scene][light]['image'], cv2.COLOR_BGR2GRAY)
            ratio_matrix[i, j] = (gray > THRESHOLD).mean() * 100

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor(BG_DARK)
    ax.set_facecolor(BG_MID)
    im = ax.imshow(ratio_matrix, cmap='YlOrRd', aspect='auto',
                   vmin=0, vmax=max(ratio_matrix.max(), 1))
    ax.set_xticks(range(len(LIGHTS)))
    ax.set_xticklabels([LIGHT_LABELS[l] for l in LIGHTS], color='white', fontsize=9)
    ax.set_yticks(range(len(SCENES)))
    ax.set_yticklabels([SCENE_LABELS[s].replace('\n', ' ') for s in SCENES],
                       color='white', fontsize=9)
    ax.tick_params(colors='white')
    for i in range(len(SCENES)):
        for j in range(len(LIGHTS)):
            val = ratio_matrix[i, j]
            color = 'black' if val > ratio_matrix.max() * 0.6 else 'white'
            ax.text(j, i, f'{val:.1f}%', ha='center', va='center',
                    color=color, fontsize=8.5, fontweight='bold')
    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label('Highlight Pixel Ratio (%)', color='white', fontsize=9)
    cbar.ax.tick_params(colors='white')
    ax.set_title(
        f'Highlight Area Ratio Matrix  (Brightness > {THRESHOLD})\n'
        f'Scene (rows) × Light Type (columns)',
        color='white', fontsize=12, pad=12
    )
    for sp in ax.spines.values():
        sp.set_color('#333344')
    plt.tight_layout()
    plt.savefig(str(OUT_DIR / 'stats' / 'highlight_ratio_heatmap.png'),
                dpi=130, bbox_inches='tight', facecolor=BG_DARK)
    plt.close()
    print("  highlight_ratio_heatmap.png")


# ============================================================
# Phase 2G: Mask coverage bar chart
# ============================================================
def make_mask_coverage(samples):
    print("\n[Phase 2G] Mask coverage bar chart...")
    coverage = {scene: [] for scene in SCENES}
    for scene in SCENES:
        for light in LIGHTS:
            mask = samples[scene][light]['mask']
            coverage[scene].append((mask > 0).mean() * 100)

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor(BG_DARK)
    ax.set_facecolor(BG_MID)
    x = np.arange(len(LIGHTS))
    bar_w = 0.10
    for i, scene in enumerate(SCENES):
        offset = (i - len(SCENES) / 2 + 0.5) * bar_w
        ax.bar(x + offset, coverage[scene], bar_w,
               label=SCENE_LABELS[scene].replace('\n', ' '),
               color=SCENE_COLORS[i], alpha=0.85,
               edgecolor=BG_DARK, linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([LIGHT_LABELS[l] for l in LIGHTS], color='white', fontsize=9)
    ax.set_ylabel('Mask Coverage (%)', color='white', fontsize=10)
    ax.set_title('Foreground Mask Coverage Rate  (Scene × Light)',
                 color='white', fontsize=12, pad=10)
    ax.tick_params(colors='white', labelsize=8)
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)
    for sp in ['bottom', 'left']:
        ax.spines[sp].set_color('#333344')
    ax.legend(fontsize=7.5, labelcolor='white', facecolor=BG_DARK,
              edgecolor='#333344', loc='upper right', ncol=2)
    ax.set_ylim(0, 105)
    ax.yaxis.grid(True, color='#333344', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(str(OUT_DIR / 'stats' / 'mask_coverage.png'),
                dpi=130, bbox_inches='tight', facecolor=BG_DARK)
    plt.close()
    print("  mask_coverage.png")


# ============================================================
# Phase 3A: Full 8×7 overview grid
# ============================================================
def make_overview_grid(samples):
    print("\n[Phase 3A] Full 8x7 overview grid...")
    n_s, n_l = len(SCENES), len(LIGHTS)
    cell = 150

    fig = plt.figure(figsize=(n_l * 2.5 + 1.8, n_s * 2.5 + 1.4))
    fig.patch.set_facecolor(BG_DARK)
    gs = gridspec.GridSpec(
        n_s + 1, n_l + 1, figure=fig,
        hspace=0.04, wspace=0.04,
        left=0.11, right=0.99, top=0.95, bottom=0.02
    )

    # Column headers (lights)
    for j, light in enumerate(LIGHTS):
        ax = fig.add_subplot(gs[0, j + 1])
        ax.set_facecolor(BG_PANEL)
        ax.text(0.5, 0.5, LIGHT_LABELS[light], ha='center', va='center',
                color='#aaddff', fontsize=8, fontweight='bold',
                transform=ax.transAxes)
        ax.axis('off')

    # Row headers (scenes)
    for i, scene in enumerate(SCENES):
        ax = fig.add_subplot(gs[i + 1, 0])
        ax.set_facecolor(BG_PANEL)
        ax.text(0.5, 0.5, SCENE_LABELS[scene], ha='center', va='center',
                color='#ffddaa', fontsize=7, fontweight='bold',
                transform=ax.transAxes)
        ax.axis('off')

    # Image cells
    for i, scene in enumerate(SCENES):
        for j, light in enumerate(LIGHTS):
            ax = fig.add_subplot(gs[i + 1, j + 1])
            img = bgr2rgb(cv2.resize(samples[scene][light]['image'], (cell, cell)))
            ax.imshow(img)
            ax.axis('off')

    # Top-left corner
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.set_facecolor(BG_DARK)
    ax0.text(0.5, 0.5, 'Scene \\ Light', ha='center', va='center',
             color='#888899', fontsize=8, transform=ax0.transAxes)
    ax0.axis('off')

    fig.suptitle(
        'National Large Metal Specular Scene — Synthetic Training Dataset Overview\n'
        '@author RussellCooper  |  8 Scenes × 7 Light Conditions = 56 Combinations',
        color='white', fontsize=11, fontweight='bold', y=0.998
    )
    fname = OUT_DIR / 'overview' / 'full_overview_8x7.png'
    plt.savefig(str(fname), dpi=150, bbox_inches='tight', facecolor=BG_DARK)
    plt.close()
    print(f"  {fname.name}")


# ============================================================
# Phase 3B: Deep 4-view per scene (4 rows × 7 lights)
# ============================================================
def make_deep_scene_views(samples):
    print("\n[Phase 3B] Deep 4-view per scene...")
    row_labels = ['Image', 'Mask', 'Edge', 'Overlay']
    for scene in SCENES:
        fig, axes = plt.subplots(4, len(LIGHTS), figsize=(len(LIGHTS) * 2.4, 10))
        fig.patch.set_facecolor(BG_DARK)
        for j, light in enumerate(LIGHTS):
            s = samples[scene][light]
            img  = bgr2rgb(s['image'])
            mask = s['mask']
            edge = s['edge']
            ov   = s['image'].copy()
            ov[s['mask'] > 0] = (ov[s['mask'] > 0] * 0.55 +
                                  np.array([0, 200, 0]) * 0.45).astype(np.uint8)
            ov[s['edge'] > 0] = [0, 0, 255]
            ov = bgr2rgb(ov)
            views = [img, mask, edge, ov]
            cmaps = [None, 'gray', 'gray', None]
            for row, (view, cmap) in enumerate(zip(views, cmaps)):
                ax = axes[row, j]
                ax.imshow(view, cmap=cmap)
                ax.axis('off')
                if row == 0:
                    ax.set_title(LIGHT_LABELS[light], color='#aaddff',
                                 fontsize=7.5, pad=2)
                if j == 0:
                    ax.set_ylabel(row_labels[row], color='#ffddaa',
                                  fontsize=8, rotation=0, labelpad=38, va='center')
        fig.suptitle(
            f'Deep 4-View: {SCENE_LABELS[scene].replace(chr(10)," ")}  '
            f'(4 views × 7 lights)',
            color='white', fontsize=11, fontweight='bold', y=1.01
        )
        plt.tight_layout(pad=0.3)
        fname = OUT_DIR / 'scene_compare' / f"{scene.value}_deep_4view.png"
        plt.savefig(str(fname), dpi=120, bbox_inches='tight', facecolor=BG_DARK)
        plt.close()
        print(f"  {fname.name}")


# ============================================================
# Phase 3C: Save metadata JSON
# ============================================================
def save_metadata(samples):
    print("\n[Phase 3C] Saving metadata JSON...")
    meta = {
        'generator': 'synth_national_scenes.py',
        'author': 'RussellCooper',
        'seed': SEED,
        'image_size': f'{IMG_H}x{IMG_W}',
        'total_combinations': len(SCENES) * len(LIGHTS),
        'scenes': {}
    }
    for s in SCENES:
        meta['scenes'][s.value] = {
            'label': SCENE_LABELS[s].replace('\n', ' '),
            'weight': NationalSceneGenerator.SCENE_WEIGHTS[s],
            'lights': {}
        }
        for l in LIGHTS:
            gray = cv2.cvtColor(samples[s][l]['image'], cv2.COLOR_BGR2GRAY)
            meta['scenes'][s.value]['lights'][l.value] = {
                'brightness_mean': round(float(gray.mean()), 2),
                'brightness_std':  round(float(gray.std()), 2),
                'highlight_ratio': round(float((gray > 220).mean() * 100), 2),
                'mask_coverage':   round(float((samples[s][l]['mask'] > 0).mean() * 100), 2),
                'edge_density':    round(float((samples[s][l]['edge'] > 0).mean() * 100), 2),
            }
    fname = OUT_DIR / 'stats' / 'dataset_metadata.json'
    with open(str(fname), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"  {fname.name}")
    return meta


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 65)
    print("  National Metal Specular Scene — Full Visualization Generator")
    print("  @author RussellCooper")
    print("=" * 65)

    setup_dirs()
    gen = NationalSceneGenerator(h=IMG_H, w=IMG_W, seed=SEED)

    samples = generate_all_samples(gen)
    make_scene_compare(samples)
    make_light_compare(samples)
    make_stats(samples)
    make_brightness_histograms(samples)
    make_edge_density(samples)
    make_highlight_ratio(samples)
    make_mask_coverage(samples)
    make_overview_grid(samples)
    make_deep_scene_views(samples)
    save_metadata(samples)

    all_files = sorted(OUT_DIR.rglob('*.png')) + sorted(OUT_DIR.rglob('*.json'))
    total_kb = sum(f.stat().st_size for f in all_files) // 1024
    print(f"\n{'='*65}")
    print(f"  Done! {len(all_files)} files, {total_kb} KB total")
    print(f"  Output: {OUT_DIR}")
    print(f"{'='*65}")
    for f in all_files:
        print(f"  {str(f.relative_to(OUT_DIR)):55s}  {f.stat().st_size//1024:4d} KB")


if __name__ == '__main__':
    main()
