"""MILP optimisation of inverter power setpoints."""

from __future__ import annotations

import os
from collections.abc import Sequence

import pyomo.environ as pyo

from config import GLPK_EXECUTABLE, INVERTER_MAX_POWER_KW, RAMP_LIMIT_KW


def optimise_setpoints(forecast: Sequence[float]) -> list[float]:
    """Minimise absolute deviation from forecast subject to inverter constraints."""
    if not forecast:
        raise ValueError("Прогноз для оптимизации пуст.")
    values = [max(0.0, float(value)) for value in forecast]
    model = pyo.ConcreteModel()
    model.time = pyo.RangeSet(0, len(values) - 1)
    model.setpoint = pyo.Var(model.time, bounds=(0, INVERTER_MAX_POWER_KW))
    model.deviation = pyo.Var(model.time, domain=pyo.NonNegativeReals)
    model.objective = pyo.Objective(expr=sum(model.deviation[t] for t in model.time), sense=pyo.minimize)
    model.absolute_deviation = pyo.ConstraintList()
    model.ramp_rate = pyo.ConstraintList()
    for time in model.time:
        model.absolute_deviation.add(model.deviation[time] >= model.setpoint[time] - values[time])
        model.absolute_deviation.add(model.deviation[time] >= values[time] - model.setpoint[time])
        if time:
            model.ramp_rate.add(model.setpoint[time] - model.setpoint[time - 1] <= RAMP_LIMIT_KW)
            model.ramp_rate.add(model.setpoint[time - 1] - model.setpoint[time] <= RAMP_LIMIT_KW)

    executable = GLPK_EXECUTABLE or os.getenv("GLPK_EXECUTABLE")
    solver = pyo.SolverFactory("glpk", executable=executable) if executable else pyo.SolverFactory("glpk")
    if not solver.available(exception_flag=False):
        raise RuntimeError("GLPK не найден. Установите GLPK или задайте путь в GLPK_EXECUTABLE.")
    result = solver.solve(model)
    if result.solver.termination_condition != pyo.TerminationCondition.optimal:
        raise RuntimeError(f"GLPK не нашёл оптимальное решение: {result.solver.termination_condition}")
    return [float(pyo.value(model.setpoint[time])) for time in model.time]
