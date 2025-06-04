# 📺 Expert Demonstration Timeline Visualization

## Single Demonstration Timeline (13.75 seconds)

```
Time:  0s    2s    3.5s   4.75s  5.75s  7.25s  9.25s  10.25s 11.75s 12.5s  13.75s
       |-----|-----|------|------|------|------|------|------|------|------|
Phase: APPR  PRE   DESC   GRASP  LIFT   MOVE   POS    PLACE  REL    RETR   DONE
Steps: [40]  [30]  [25]   [20]   [30]   [40]   [20]   [30]   [15]   [25]
Grip:  Open  Open  Open   Close  Close  Close  Close  Close  Open   Open

Robot: 🤖    🤖    🤖     🤖     🤖↑    🤖→    🤖     🤖↓    🤖     🤖←
Block:       📦           📦     📦     📦     📦     📦            
                                        ↑      →             ↓
```

## Key Phases Explained:

### 1. **APPROACH** (0-2s)
- Robot moves from home position toward red block
- Large joint movements to get close
- Gripper stays open

### 2. **PRE_GRASP** (2-3.5s) 
- Fine positioning directly above block
- Small corrections for alignment
- Prepares for descent

### 3. **DESCEND** (3.5-4.75s)
- Controlled lowering to grasp height
- Maintains X-Y alignment
- Stops just above block

### 4. **GRASP** (4.75-5.75s)
- No joint movement
- Gripper closes on block
- Tactile sensors activate

### 5. **LIFT** (5.75-7.25s)
- Vertical movement upward
- Block lifted 5-10mm
- Success checkpoint!

### 6. **MOVE_TO_BLUE** (7.25-9.25s)
- Lateral movement toward blue block
- Maintains lift height
- Carries red block

### 7. **POSITION** (9.25-10.25s)
- Fine positioning above blue block
- Alignment corrections
- Prepares for placement

### 8. **PLACE** (10.25-11.75s)
- Controlled descent
- Places red on blue
- Maintains grip

### 9. **RELEASE** (11.75-12.5s)
- Opens gripper
- Releases red block
- Task complete!

### 10. **RETREAT** (12.5-13.75s)
- Moves away from stack
- Returns to safe position
- Demo finished

## Data Collection Summary:

```
For 50 demonstrations:
- Total time: ~11.5 minutes
- Success rate: ~90%
- Data size: ~50MB
- Frames (if video): ~20,000
```

## Quick Reference:

| Phase | Duration | Key Action | Gripper | Success Metric |
|-------|----------|------------|---------|----------------|
| Approach | 2.0s | Move to block | Open | Distance < 5cm |
| Pre-grasp | 1.5s | Align above | Open | XY error < 2cm |
| Descend | 1.25s | Lower to block | Open | Z = block height |
| Grasp | 1.0s | Close gripper | →Closed | Tactile > 0 |
| Lift | 1.5s | Raise block | Closed | Height > 5mm |
| Move | 2.0s | Go to blue | Closed | Maintain grip |
| Position | 1.0s | Align above blue | Closed | XY aligned |
| Place | 1.5s | Lower onto blue | Closed | Contact made |
| Release | 0.75s | Open gripper | →Open | Block stable |
| Retreat | 1.25s | Move away | Open | Clear of blocks |