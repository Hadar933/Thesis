from Forces import parse_forces


def get_forces(date: str, show_results: False):
    # TODO: generalize path
    df, header = parse_forces.read_forces_csv(f'Forces\\experiments\\{date}\\trigger_test2.csv')
    if show_results:
        parse_forces.plot_forces(df, header)
    return df
