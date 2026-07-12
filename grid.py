def create_factorial_grid(m_options = 2,
                           nu_options = 2,
                             q_options = 2,
                               omega_options = 2,
                                 lamba_c_options = 2,
                                   sigma_c_options = 2,
                                     iter_options = 2,
                                       burnin_options = 2):
    m_range = [20, 500]
    nu_range = [1,10]
    q_range = [0.6, 0.999]
    omega_range = [1, 7]
    lamba_c_range = [1,50]
    sigma_c_range = [0.1, 3]
    iter_range = [500,100000]
    burnin_range = [0.01, 0.5]

    import csv
    from pathlib import Path

    import numpy as np
    m_grid = np.round(np.linspace(m_range[0], m_range[1], m_options)).astype(int)
    nu_grid = np.round(np.linspace(nu_range[0], nu_range[1], nu_options)).astype(int)
    q_grid = np.linspace(q_range[0], q_range[1], q_options)
    omega_grid = np.round(np.linspace(omega_range[0], omega_range[1], omega_options)).astype(int)
    lamba_c_grid = np.round(np.linspace(lamba_c_range[0], lamba_c_range[1], lamba_c_options)).astype(int)
    sigma_c_grid = np.linspace(sigma_c_range[0], sigma_c_range[1], sigma_c_options)
    iter_grid = np.round(np.linspace(iter_range[0], iter_range[1], iter_options)).astype(int)
    burnin_grid = np.round(np.linspace(burnin_range[0], burnin_range[1], burnin_options)).astype(int)

    output_path = Path(__file__).with_name("grid.csv")
    with output_path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        for m in m_grid:
            for nu in nu_grid:
                for q in q_grid:
                    for omega in omega_grid:
                        for lamba_c in lamba_c_grid:
                            for sigma_c in sigma_c_grid:
                                for iter in iter_grid:
                                    for burnin in burnin_grid:
                                        writer.writerow([m, nu, q, omega, lamba_c, sigma_c, iter, burnin])

    return output_path

create_factorial_grid()