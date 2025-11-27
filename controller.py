import numpy as np
from numpy.typing import ArrayLike

from racetrack import RaceTrack

# parameters for fine tuning
controlDt = 0.1  # timestep for the controller
velKp = 41  # velocity proportional gain
steerKp = 12.5  # steering proportional gain
steerKi = 16.5  # steering integral gain
steerKd = 0.145  # steering derivative gain

baseLookaheadDist = 25  # default distance to look ahead on the path
speedPreviewDistance = 15  # distance ahead to analyze for velocity planning
pathBlendRatio = 0.52  # mixing factor: 0.52 means slightly favor raceline over centerline

prevSteeringError = 0.0  # previous steering error for derivative term
steeringIntegral = 0.0  # accumulated steering error for integral term
maxIntegralValue = 0.25  # clamp integral to prevent windup


def computeCurvatureMetric(path, widths, focusIndex, analysisWindow=10):
    # compute how sharp the turn is at a given point on the path
    windowHalf = analysisWindow // 2
    beginIdx = max(focusIndex - windowHalf, 1)
    endIdx = min(focusIndex + windowHalf, len(path) - 3)
    
    turnMetrics = []
    pathSegmentLengths = []
    
    for idx in range(beginIdx, endIdx + 1):
        # look at three points to estimate curvature
        p0 = path[idx - 2]
        p1 = path[idx]
        p2 = path[idx + 2]
        
        direction1 = p1 - p0
        direction2 = p2 - p1
        d1Mag = np.linalg.norm(direction1)
        d2Mag = np.linalg.norm(direction2)
        
        if d1Mag < 1e-6 or d2Mag < 1e-6:
            continue  # skip if points are too close
        
        # figure out the angle between direction vectors using dot product
        dotProduct = np.dot(direction1, direction2)
        cosVal = np.clip(dotProduct / (d1Mag * d2Mag), -1.0, 1.0)
        turnAngle = np.arccos(cosVal)
        
        # make it signed so we know if it's a left or right turn
        crossVal = np.cross(direction1, direction2)
        if crossVal < 0:
            turnAngle = -turnAngle
        
        # convert angle to steering value, cap at 0.77 rad
        if abs(turnAngle) < 0.77:
            steeringValue = np.tan(turnAngle * 2.0) / 2.0
        else:
            steeringValue = 100.0  # really sharp turn
        
        # normalize by track width
        steeringValue = (steeringValue / widths[idx]) * 10.0
        
        turnMetrics.append(steeringValue)
        pathSegmentLengths.append(d1Mag)
    
    if len(turnMetrics) < 2:
        return 0.0  # not enough data
    
    turnMetrics = np.array(turnMetrics)
    pathSegmentLengths = np.array(pathSegmentLengths)
    
    # compute rate of change of steering
    metricDifferences = np.diff(turnMetrics)
    metricDifferences = (metricDifferences + np.pi) % (2 * np.pi) - np.pi
    
    meanSegmentLength = (pathSegmentLengths[1:] + pathSegmentLengths[:-1]) * 0.5
    changeRate = np.abs(metricDifferences) / np.maximum(meanSegmentLength, 1e-6)
    
    # take top 5 highest change rates and average them
    sortedRates = np.sort(changeRate)
    topFive = sortedRates[-5:]
    meanIntensity = np.mean(topFive)
    
    return meanIntensity


def findPathIndexAtDistance(path, initialIndex, targetDistance):
    # walk along the path from initialIndex until we've traveled targetDistance
    pathLength = len(path)
    distanceSum = 0.0
    idx = initialIndex
    loopCount = 0
    
    while distanceSum < targetDistance and loopCount < min(pathLength, 1000):
        followingIdx = (idx + 1) % pathLength  # wrap around if needed
        stepSize = np.linalg.norm(path[followingIdx] - path[idx])
        
        if stepSize == 0:
            break  # avoid infinite loop
        
        distanceSum += stepSize
        idx = followingIdx
        loopCount += 1
    
    return idx


def adjustPathSize(originalPath, newCount):
    # change path to have newCount points by interpolating between existing points
    if len(originalPath) == newCount:
        return originalPath.copy()
    
    oldIndices = np.arange(len(originalPath))
    newIndices = np.linspace(0, len(originalPath) - 1, newCount)
    
    xNew = np.interp(newIndices, oldIndices, originalPath[:, 0])
    yNew = np.interp(newIndices, oldIndices, originalPath[:, 1])
    
    return np.column_stack((xNew, yNew))


def controller(state, parameters, racetrack):
    # main controller - computes desired steering and velocity
    stateArr = np.asarray(state, dtype=float)
    paramsArr = np.asarray(parameters, dtype=float)
    
    carPosition = stateArr[:2]
    currentHeading = stateArr[4]
    
    # blend raceline and centerline if raceline exists
    if hasattr(racetrack, 'raceline') and racetrack.raceline is not None:
        racePath = racetrack.raceline
        centerPath = racetrack.centerline
        
        if len(racePath) != len(centerPath):
            centerPath = adjustPathSize(centerPath, len(racePath))
        
        blendedPath = pathBlendRatio * racePath + (1.0 - pathBlendRatio) * centerPath
    else:
        blendedPath = racetrack.centerline
    
    boundaryWidths = np.linalg.norm(racetrack.right_boundary - racetrack.left_boundary, axis=1)
    
    # find closest point on path
    positionDeltas = blendedPath - carPosition
    squaredDistances = np.sum(positionDeltas * positionDeltas, axis=1)
    closestPathIdx = int(np.argmin(squaredDistances))
    
    # look ahead for speed planning
    velocityLookaheadIdx = findPathIndexAtDistance(blendedPath, closestPathIdx, baseLookaheadDist)
    curvatureAnalysisIdx = velocityLookaheadIdx + speedPreviewDistance
    
    turnIntensity = 2.0 * computeCurvatureMetric(blendedPath, boundaryWidths, curvatureAnalysisIdx, 40)
    
    # also check immediate curvature for faster response
    immediateCurvatureIdx = findPathIndexAtDistance(blendedPath, closestPathIdx, 8.0)
    immediateTurnIntensity = 2.0 * computeCurvatureMetric(blendedPath, boundaryWidths, immediateCurvatureIdx, 20)
    
    maxCurvature = max(abs(turnIntensity), abs(immediateTurnIntensity))
    
    # compute desired velocity based on curvature
    maxLateralAccel = 44.5
    curvatureSafe = max(maxCurvature, 0.001)
    velocityFromCurvature = np.sqrt(maxLateralAccel / curvatureSafe)
    if curvatureSafe > 0.1:
        # extra penalty for really tight corners
        curvaturePenalty = 1.0 + 2.0 * (curvatureSafe - 0.1)
        velocityFromCurvature /= curvaturePenalty
    desiredVelocity = min(velocityFromCurvature, paramsArr[5])
    
    # compute current curvature for lookahead adjustment
    currentTurnIntensity = computeCurvatureMetric(blendedPath, boundaryWidths, closestPathIdx, 20)
    currentCurvature = 2.0 * currentTurnIntensity
    
    # adjust lookahead based on velocity
    velocityReduction = int((100.0 - desiredVelocity) / 8.2)
    baseSteeringLookahead = baseLookaheadDist - velocityReduction
    
    # reduce lookahead more aggressively for tight corners
    if currentCurvature > 0.15:
        # very tight corner - use very short lookahead
        curvatureLookaheadReduction = 12.0 + 8.0 * min((currentCurvature - 0.15) / 0.1, 1.0)
        minLookahead = max(8.0, 20.0 - curvatureLookaheadReduction)
    elif currentCurvature > 0.08:
        # medium corner - moderate reduction
        curvatureLookaheadReduction = 5.0 + 5.0 * ((currentCurvature - 0.08) / 0.07)
        minLookahead = max(10.0, 18.0 - curvatureLookaheadReduction)
    else:
        # gentle corner or straight - use normal lookahead
        curvatureLookaheadReduction = 3.0 * (currentCurvature / 0.08)
        minLookahead = max(12.0, 15.0 - curvatureLookaheadReduction)
    
    # extra reduction for extremely tight corners
    if currentCurvature > 0.2:
        minLookahead = max(6.0, minLookahead - 3.0)
    
    steeringLookaheadDist = max(baseSteeringLookahead, minLookahead)
    steeringLookaheadIdx = findPathIndexAtDistance(
        blendedPath, closestPathIdx, steeringLookaheadDist
    )
    
    # pure pursuit steering
    targetPoint = blendedPath[steeringLookaheadIdx]
    pointDirection = targetPoint - carPosition
    desiredHeading = np.arctan2(pointDirection[1], pointDirection[0])
    
    headingDiff = desiredHeading - currentHeading
    headingDiff = (headingDiff + np.pi) % (2 * np.pi) - np.pi  # normalize to [-pi, pi]
    
    vehicleWheelbase = paramsArr[0]
    lookaheadMagnitude = np.linalg.norm(pointDirection)
    effectiveCurvature = 2.0 * (np.sin(headingDiff) / max(lookaheadMagnitude, 0.001))
    
    steeringCommand = np.arctan(vehicleWheelbase * effectiveCurvature)
    steeringCommand = np.clip(steeringCommand, paramsArr[1], paramsArr[4])
    
    return np.array([steeringCommand, desiredVelocity], float)


def lower_controller(state, desired, parameters):
    # low level controller - converts desired steering/velocity to control inputs (steering rate, acceleration)
    global prevSteeringError, steeringIntegral
    
    stateArr = np.asarray(state, dtype=float)
    desiredArr = np.asarray(desired, dtype=float)
    paramsArr = np.asarray(parameters, dtype=float)
    
    # compute steering error
    deltaError = (desiredArr[0] - stateArr[2] + np.pi) % (2 * np.pi) - np.pi
    deltaErrorDerivative = (deltaError - prevSteeringError) / controlDt
    
    # anti-windup: decay integral when error is large (during sharp turns)
    if abs(deltaError) > 0.3:
        steeringIntegral *= 0.98
    else:
        steeringIntegral += deltaError * controlDt
    
    # clamp integral to prevent windup
    steeringIntegral = np.clip(steeringIntegral, -maxIntegralValue, maxIntegralValue)
    
    prevSteeringError = deltaError
    
    # pid control for steering rate
    steeringRateCmd = (
        steerKp * deltaError +
        steerKi * steeringIntegral +
        steerKd * deltaErrorDerivative
    )
    steeringRateCmd = np.clip(steeringRateCmd, paramsArr[7], paramsArr[9])
    
    # proportional control for velocity
    vError = desiredArr[1] - stateArr[3]
    accelCmd = velKp * vError
    accelCmd = np.clip(accelCmd, paramsArr[8], paramsArr[10])
    
    return np.array([steeringRateCmd, accelCmd], float)