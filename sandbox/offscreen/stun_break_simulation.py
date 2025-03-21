import shutil
from pathlib import Path
from typing import Dict

import numpy as np

import pooltool as pt
import pooltool.constants as c
from pooltool.ani.camera import camera_states
from pooltool.ani.image import ImageZip, image_stack
from pooltool.ani.image.interface import FrameStepper
from pooltool.game.datatypes import GameType
from pooltool.layouts import (
    DEFAULT_STANDARD_BALLSET,
    BallPos,
    Dir,
    Jump,
    ball_cluster_blueprint,
    generate_layout,
)
from pooltool.objects.ball.datatypes import Ball, BallParams
from pooltool.objects.table.datatypes import Table


def setup_eightball_stun_break(
    table: Table,
    ball_params: BallParams = BallParams.default(game_type=GameType.EIGHTBALL),
    **kwargs,
) -> Dict[str, Ball]:
    stripes = {"9", "10", "11", "12", "13", "14", "15"}
    solids = {"1", "2", "3", "4", "5", "6", "7"}

    blueprint = ball_cluster_blueprint(
        seed=BallPos([], (0.5, 0.75), solids),
        jump_sequence=[
            (Jump.DOWN(), {"cue"}),
            # row 2
            ([Dir.UP, Dir.UPLEFT], stripes),
            (Jump.RIGHT(), solids),
            # row 3
            (Jump.UPRIGHT(), stripes),
            (Jump.LEFT(), {"8"}),
            (Jump.LEFT(), solids),
            # row 4
            (Jump.UPLEFT(), stripes),
            (Jump.RIGHT(), solids),
            (Jump.RIGHT(), stripes),
            (Jump.RIGHT(), solids),
            # row 5
            (Jump.UPRIGHT(), stripes),
            (Jump.LEFT(), solids),
            (Jump.LEFT(), stripes),
            (Jump.LEFT(), stripes),
            (Jump.LEFT(), solids),
        ],
    )

    return generate_layout(blueprint, table, ballset=DEFAULT_STANDARD_BALLSET, **kwargs)


def main(args):
    stepper = FrameStepper()

    if args.seed:
        np.random.seed(args.seed)

    system = pt.System(
        cue=pt.Cue(cue_ball_id="cue"),
        table=(table := pt.Table.default()),
        balls=setup_eightball_stun_break(table, spacing_factor=args.spacing_factor),
    )

    cueball = system.balls["cue"]
    cueball.state.s = c.sliding
    cueball.state.rvw[1][1] = args.V0  # v_y

    # Evolve the shot
    pt.simulate(system, inplace=True)

    # Make an dump dir
    path = Path(__file__).parent / "images_out"
    if path.exists():
        shutil.rmtree(path)
    path.mkdir()

    for camera_state in [
        "7_foot_overhead",
        "7_foot_overhead_zoom",
        "7_foot_overhead_footspot",
        "rack",
    ]:
        exporter = ImageZip(path / f"{camera_state}.zip", ext="png")

        imgs = image_stack(
            system=system,
            interface=stepper,
            size=(1920, 1080),
            fps=250,
            camera_state=camera_states[camera_state],
            show_hud=False,
            gray=False,
        )
        exporter.save(imgs)

        # Verify the images can be read back
        read_from_disk = exporter.read(exporter.path)
        assert np.array_equal(imgs, read_from_disk)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--spacing-factor",
        type=float,
        default=1e-3,
        help="What fraction of the ball radius should each ball be randomly separated "
        "by in the rack?",
    )
    ap.add_argument(
        "--V0",
        type=float,
        default=8.89,
        help="With what speed the cue ball should impact the rack",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Provide a random seed if you want reproducible results",
    )

    args = ap.parse_args()

    main(args)
