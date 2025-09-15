from domiknows.program.lossprogram import LossProgram

def build_program(graph, sensors, learners, constraints):
    program = LossProgram(graph)
    program.addSensor(sensors["header"], concept="Header")
    program.addSensor(sensors["body"], concept="Body")
    program.addLearner(learners["model1"], concept="Model1Label")
    program.addLearner(learners["model2"], concept="Model2Label")
    program.addConstraint(constraints["consistency"])
    return program