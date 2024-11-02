from textwrap import wrap

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def create_category_plot(data_dict, title, filename):
    """Create a horizontal bar plot for category distribution.

    Args:
        data_dict (dict): A dictionary with categories as keys and counts as values.
        title (str): The title of the plot.
        filename (str): The filename to save the plot.
    """
    df = pd.DataFrame(list(data_dict.items()), columns=["Category", "Count"])
    df = df.sort_values("Count", ascending=True)

    plt.figure(figsize=(15, 10))
    bars = plt.barh(df["Category"], df["Count"])

    plt.title(title, pad=20, fontsize=14, fontweight="bold")
    plt.xlabel("Number of Cases", fontsize=12)

    for bar in bars:
        width = bar.get_width()
        plt.text(
            width,
            bar.get_y() + bar.get_height() / 2,
            f"{int(width):,}",
            ha="left",
            va="center",
            fontsize=10,
        )

    plt.yticks(
        range(len(df["Category"])),
        ["\n".join(wrap(label, 30)) for label in df["Category"]],
        fontsize=10,
    )

    plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight", dpi=300)
    plt.close()


def create_treemap(data_dict, title, filename):
    """Create a treemap visualization using matplotlib.

    Args:
        data_dict (dict): A dictionary with categories as keys and counts as values.
        title (str): The title of the treemap.
        filename (str): The filename to save the treemap.
    """
    import matplotlib.patches as patches

    df = pd.DataFrame(list(data_dict.items()), columns=["Category", "Count"])
    total = df["Count"].sum()
    df["Percentage"] = df["Count"] / total * 100
    df = df.sort_values("Count", ascending=False)

    plt.figure(figsize=(20, 10))

    def get_rect_coords(areas):
        coords = []
        current_x = 0
        current_y = 0
        row_height = 0
        width = 100

        for area in areas:
            rect_width = (area / 100) * width
            if current_x + rect_width > width:
                current_x = 0
                current_y += row_height
                row_height = 0

            rect_height = rect_width
            row_height = max(row_height, rect_height)

            coords.append((current_x, current_y, rect_width, rect_height))
            current_x += rect_width

        return coords

    coords = get_rect_coords(df["Percentage"])
    colors = plt.cm.viridis(np.linspace(0, 1, len(df)))

    ax = plt.gca()
    for (x, y, w, h), color, (_, row) in zip(coords, colors, df.iterrows()):
        rect = patches.Rectangle((x, y), w, h, facecolor=color, edgecolor="white")
        ax.add_patch(rect)

        if row["Percentage"] > 2:
            plt.text(
                x + w / 2,
                y + h / 2,
                f"{row['Category']}\n{row['Count']:,}\n({row['Percentage']:.1f}%)",
                ha="center",
                va="center",
                wrap=True,
            )

    plt.title(title, pad=20, fontsize=14, fontweight="bold")
    plt.axis("equal")
    plt.axis("off")
    plt.savefig(filename, bbox_inches="tight", dpi=300)
    plt.close()


def main():
    """Main function to create visualizations for cybercrime data."""
    category_data = {
        "Online Financial Fraud": 56718,
        "Online and Social Media Related Crime": 11877,
        "Any Other Cyber Crime": 10727,
        "Cyber Attack/ Dependent Crimes": 3608,
        "RapeGang Rape RGRSexually Abusive Content": 2816,
        "Sexually Obscene material": 1819,
        "Hacking  Damage to computercomputer system etc": 1682,
        "Sexually Explicit Act": 1530,
        "Cryptocurrency Crime": 477,
        "Online Gambling  Betting": 438,
        "Child Pornography CPChild Sexual Abuse Material CSAM": 374,
        "Online Cyber Trafficking": 180,
        "Cyber Terrorism": 160,
        "Ransomware": 56,
        "Report Unlawful Content": 1,
    }

    sub_category_data = {
        "UPI Related Frauds": 26479,
        "Other": 10727,
        "DebitCredit Card FraudSim Swap Fraud": 10690,
        "Internet Banking Related Fraud": 8825,
        "Unknown": 6539,
        "Fraud CallVishing": 5709,
        "Cyber Bullying  Stalking  Sexting": 4004,
        "EWallet Related Fraud": 3981,
        "FakeImpersonating Profile": 2246,
        "Profile Hacking Identity Theft": 2025,
        "Cheating by Impersonation": 1949,
        "Unauthorised AccessData Breach": 1094,
        "Online Job Fraud": 901,
        "DematDepository Fraud": 747,
        "Tampering with computer source documents": 567,
    }

    create_category_plot(
        category_data,
        "Distribution of Cybercrime Categories",
        "cybercrime_categories.png",
    )

    create_category_plot(
        sub_category_data,
        "Top 15 Cybercrime Sub-Categories",
        "cybercrime_subcategories.png",
    )

    create_treemap(
        category_data, "Cybercrime Categories Treemap", "cybercrime_treemap.png"
    )


if __name__ == "__main__":
    main()
