import shutil
from pathlib import Path
from typing import Dict

import numpy as np

import pooltool as pt
import pooltool.constants as c
from pooltool.ani.camera import CameraState, camera_states
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

__camera_states = camera_states | {
    path.stem: CameraState.from_json(path)
    for path in (Path(__file__).parent / "cameras").glob("*.json")
}


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


def setup_frozen_balls(
    table: Table,
    ball_params: BallParams = BallParams.default(game_type=GameType.EIGHTBALL),
    **kwargs,
) -> Dict[str, Ball]:
    blueprint = ball_cluster_blueprint(
        seed=BallPos([], (0.5, 0.75), {"1"}),
        jump_sequence=[
            (Jump.DOWN(), {"cue"}),
            ([Dir.UP, Dir.UP], {"2"}),
        ],
    )
    return generate_layout(blueprint, table, ballset=DEFAULT_STANDARD_BALLSET, **kwargs)


def main(args):
    if args.seed is not None:
        np.random.seed(args.seed)

    table = pt.Table.default()

    kwargs = {}
    if args.spacing_factor is not None:
        kwargs["spacing_factor"] = args.spacing_factor
    balls = setup_frozen_balls(table, **kwargs)

    system = pt.System(cue=pt.Cue(cue_ball_id="cue"), table=table, balls=balls)

    cueball = system.balls["cue"]
    cueball.state.s = c.sliding
    cueball.state.rvw[1][1] = args.V0  # v_y

    # Evolve the shot
    pt.simulate(system, inplace=True)

    if not args.render_images:
        camera_state = __camera_states["footspot"]
        pt.show(system, camera_state=camera_state)
    else:
        # Make an dump dir
        path = Path(args.output_dir)
        if path.exists():
            shutil.rmtree(path)
        path.mkdir()

        stepper = FrameStepper()

        for camera_state in ["footspot"]:
            exporter = ImageZip(path / f"{camera_state}.zip", ext="png")

            imgs = image_stack(
                system=system,
                interface=stepper,
                size=(1600, 900),
                fps=250,
                camera_state=__camera_states[camera_state],
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
        help="fraction of the ball radius by which each ball should be randomly"
        "separated in the rack",
    )
    ap.add_argument(
        "--V0",
        type=float,
        default=8.89,
        help="speed the cue ball should impact the rack",
    )
    ap.add_argument(
        "--seed",
        type=int,
        help="seed for reproducible random results",
    )
    ap.add_argument(
        "--render-images",
        action=argparse.BooleanOptionalAction,
        help="whether or not to render images",
    )
    ap.add_argument(
        "--output-dir",
        type=str,
        default="images_out",
        help="directory to put output images",
    )

    args = ap.parse_args()

    main(args)
