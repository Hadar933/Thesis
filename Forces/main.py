from Forces import parse_forces


def get_forces(
        csv_filename: str,
        show_force_results: bool,
        header_row_count: int
):
    corresponding_force_csv_name = f"Forces[{csv_filename.split('[')[1].split(']')[0]}]"
    df, header = parse_forces.read_forces_csv(
        csv_filename=corresponding_force_csv_name,
        header_row_count=header_row_count
    )
    if show_force_results:
        parse_forces.plot_forces(df, header)
    return df
