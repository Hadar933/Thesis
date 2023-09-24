from Forces import parse_forces
from loguru import logger


def get_forces(
        exp_date,
        parent_dirname: str,
        photos_sub_dirname: str,
        show_force_results: bool,
        header_row_count: int
):
    forces_csv = f"{parent_dirname}\\experiments\\{exp_date}\\forces\\{photos_sub_dirname.replace('Photos', 'Forces')}.csv"
    logger.info(f'Extracting Forces from {forces_csv}...')
    df, header = parse_forces.read_forces_csv(
        csv_filename=forces_csv,
        header_row_count=header_row_count
    )
    if show_force_results:
        parse_forces.plot_forces(df, header)
    return df
