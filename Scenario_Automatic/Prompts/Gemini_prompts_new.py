ER_PROMPT_TEMPLATE = """Analyze this microscopy image to identify Endoplasmic Reticulum (ER) structures for SAM3 segmentation.

TASK: Provide detailed prompts for SAM3 (Segment Anything Model 3) to accurately segment all ER regions.

WHAT TO DETECT:
{what_to_detect}

PROMPT REQUIREMENTS:

1. positive_boxes: Bounding boxes [ymin, xmin, ymax, xmax] around distinct ER regions
   - FIRST: Analyze the image and estimate how many distinct ER regions are visible
   - THEN: Provide one bounding box for each distinct, well-separated ER region
   - Cover all major ER structures that are clearly visible
   - Make boxes tight around visible ER patterns
   - AVOID including other organelles (mitochondria, nuclei, vesicles, Golgi apparatus)
   - Focus on regions with clear ER morphology only
   - Each box should contain primarily ER, not mixed structures
   - Number of boxes should match the number of distinct ER regions you identify

2. positive_points: [y, x] coordinates INSIDE clear ER structures (8-15 points)
   - Place points on brightest/clearest ER regions
   - Distribute across different ER areas
   - Focus on tubules, cisternae centers, and dense ER regions
   - MORE positive points = better segmentation

3. negative_points: [y, x] coordinates in NON-ER regions (10-20 points)
   - Place in nuclei (dark circular regions)
   - Place in mitochondria (small dark organelles)
   - Place in Golgi apparatus
   - Place in vesicles and vacuoles
   - Place in background/cytoplasm (areas without ER)
   - Place near ER boundaries to refine edges
   - MORE negative points = cleaner segmentation, fewer false positives

COORDINATE FORMAT:
- All coordinates normalized to [0, 1000] range
- Based on image dimensions: [0, 1000] maps to [0, width] and [0, height]

OUTPUT FORMAT:
Return ONLY valid JSON (no markdown, no backticks, no extra text):
{{
  "positive_boxes": [[ymin, xmin, ymax, xmax], ...],
  "positive_points": [[y, x], ...],
  "negative_points": [[y, x], ...]
}}

CRITICAL: 
- First assess the image to determine the appropriate number of bounding boxes
- Provide MORE points (8-15 positive, 10-20 negative) for accurate SAM3 segmentation
- Ensure boxes exclude other organelles to avoid contamination
- Adapt the number of boxes based on image complexity and ER distribution"""

# Default "what_to_detect" for initial prompt
ER_DEFAULT_DETECTION = """- Endoplasmic Reticulum: interconnected network of flattened sacs (cisternae) and tubular structures
- Look for reticular patterns, branching tubules, and sheet-like structures"""

ER_FEEDBACK ="""You are an expert electron microscopist evaluating ER segmentation quality.

IMAGE LAYOUT:
- LEFT: Original electron microscopy image showing Endoplasmic Reticulum structures
- RIGHT: Current segmentation result (WHITE = detected as ER, BLACK = background)

TASK: Evaluate segmentation quality by visual comparison of LEFT vs RIGHT images.

EVALUATION CRITERIA:

Carefully examine BOTH images side-by-side. For each criterion below, compare what you see in the LEFT image (ground truth ER) with what is captured in the RIGHT image (segmentation).

1. MORPHOLOGICAL COVERAGE (Weight: 25%)
   Look at the LEFT image: How many distinct ER structures do you see?
   - Count major cisternae (flattened sacs)
   - Count tubular networks
   - Identify nuclear envelope regions
   
   Now look at RIGHT: What percentage of these structures are captured as white pixels?
   
   Score 0.0-1.0:
   - 1.0: All visible ER structures captured
   - 0.75: Most major structures captured, some minor tubules missing
   - 0.5: About half the structures captured
   - 0.25: Only a few structures captured
   - 0.0: Almost nothing captured
   
   CRITICAL: Be strict. If you see 4 cisternae in LEFT but only 2 are white in RIGHT, score ~0.5, not higher.

2. BOUNDARY ACCURACY (Weight: 25%)
   For each ER structure visible in LEFT, examine its boundaries.
   
   In RIGHT, check if white pixels:
   - Stop exactly at membrane edges (good) or extend into cytoplasm (bad)
   - Cover the full membrane width (good) or only partial coverage (bad)
   - Follow smooth membrane contours (good) or have jagged irregular edges (bad)
   
   Score 0.0-1.0:
   - 1.0: Boundaries precisely match membrane edges in all structures
   - 0.75: Most boundaries accurate, slight looseness/tightness in some areas
   - 0.5: Noticeable boundary errors - some regions too broad or too narrow
   - 0.25: Significant boundary problems throughout
   - 0.0: Boundaries very poor
   
   CRITICAL: Even if coverage is good, poor boundaries should lower this score significantly.

3. SPATIAL ACCURACY (Weight: 20%)
   Compare the spatial extent and distribution in LEFT vs RIGHT.
   
   For each ER structure in LEFT:
   - Is its FULL LENGTH captured in RIGHT (top to bottom, end to end)?
   - Are thin connecting tubules between cisternae included?
   - Are peripheral regions at image edges captured?
   
   Score 0.0-1.0:
   - 1.0: Complete spatial extent of all structures captured
   - 0.75: Most structures captured fully, some truncated regions
   - 0.5: Many structures only partially captured (e.g., only middle portions)
   - 0.25: Most structures severely truncated
   - 0.0: Only tiny fragments of structures captured

4. CONTINUITY PRESERVATION (Weight: 15%)
   Look at LEFT: ER forms continuous elongated structures.
   
   Look at RIGHT: Are these continuous structures preserved, or broken into fragments?
   
   For each continuous ER sheet/tubule in LEFT:
   - Count how many disconnected white regions it becomes in RIGHT
   - A single sheet should be one continuous white region, not 3-4 pieces
   
   Score 0.0-1.0:
   - 1.0: All continuous structures remain continuous
   - 0.75: Minor fragmentation (1-2 breaks in otherwise continuous structures)
   - 0.5: Moderate fragmentation (structures broken into 2-3 pieces)
   - 0.25: Heavy fragmentation (structures broken into many small pieces)
   - 0.0: Severe fragmentation

5. SPECIFICITY (Weight: 15%)
   Look at areas in LEFT that are clearly NOT ER (mitochondria, Golgi, background, nuclei).
   
   In RIGHT, are these non-ER regions correctly BLACK?
   
   Check for white pixels in:
   - Oval mitochondria with internal cristae
   - Background cytoplasm
   - Dark nuclear regions
   - Round vesicles
   
   Score 0.0-1.0:
   - 1.0: Only ER is white, all non-ER is black
   - 0.75: Minor false positives (small scattered white pixels in background)
   - 0.5: Moderate false positives (some non-ER structures incorrectly white)
   - 0.25: Major false positives (large non-ER regions white)
   - 0.0: Severe false positives throughout

SCORING METHODOLOGY:

Step 1: Score each criterion independently (0.0-1.0)
Step 2: Calculate weighted average:
   quality_score = (morphological_coverage × 0.25) + 
                   (boundary_accuracy × 0.25) + 
                   (spatial_accuracy × 0.20) + 
                   (continuity × 0.15) + 
                   (specificity × 0.15)

Step 3: Provide visual guidance for improvement

IMPORTANT CALIBRATION GUIDELINES:
- A score of 0.8+ means segmentation is genuinely excellent (>80% IoU equivalent)
- A score of 0.5-0.7 means partial success with significant room for improvement
- A score below 0.5 means poor performance


OUTPUT FORMAT (JSON only, no markdown, no backticks):
{{
  "quality_score": <float 0.0-1.0, calculated weighted average>,
  "criteria_scores": {{
    "morphological_coverage": <float 0.0-1.0>,
    "boundary_accuracy": <float 0.0-1.0>,
    "spatial_accuracy": <float 0.0-1.0>,
    "continuity": <float 0.0-1.0>,
    "specificity": <float 0.0-1.0>
  }},
  "visual_guidance": "3-5 sentences describing ER detection guidance:

Write balanced guidance that acknowledges what's working AND what needs improvement.

Structure your response based on quality_score:

HIGH QUALITY (0.7-1.0):
Start by describing what the segmentation does well, then suggest refinements.
Example: 'ER appears as thin parallel vertical sheets with smooth continuous membranes. Current detection successfully captures the main cisternal structures with accurate boundary precision. Successfully maintaining continuity of elongated sheets. To further refine: include thin tubular extensions that connect between main cisternae, particularly visible in the upper regions.'

MODERATE QUALITY (0.5-0.7):
Balance strengths and weaknesses, prioritize key improvements.
Example: 'ER appears as elongated cisternae with reticular connections. Successfully detecting major cisternal sheets with reasonable coverage. However, boundaries are too loose - should tightly follow membrane edges rather than broad regions. Also missing thin tubular networks. Priority: tighten boundaries and include fine tubular structures.'

LOW QUALITY (<0.5):
Briefly note any partial success, then focus on major improvements needed.
Example: 'ER appears as thin dark linear structures forming parallel sheets. Current approach captures only central portions of main cisternae. Need to: include complete spatial extent from top to bottom, add thin tubular connections, maintain continuity without fragmentation, and tighten boundaries to match narrow membrane width.'

ALWAYS include:
1. Brief description of what ER looks like in this image
2. What aspect(s) are working well (if any)
3. What needs improvement (prioritized)
4. Specific actionable guidance using 'Include...', 'Detect...', 'Maintain...', 'Avoid...'

Focus on being constructive and specific to THIS image."
}}

Provide honest, balanced assessment that helps improve detection while acknowledging successes.
"""

MITO_PROMPT_TEMPLATE = """Analyze this microscopy image to identify Mitochondria structures for SAM3 segmentation.

TASK: Provide detailed prompts for SAM3 (Segment Anything Model 3) to accurately segment all mitochondria regions.

WHAT TO DETECT:
{what_to_detect}

PROMPT REQUIREMENTS:
1. positive_boxes: Bounding boxes [ymin, xmin, ymax, xmax] around distinct mitochondria
   - FIRST: Analyze the image and estimate how many distinct mitochondria are visible
   - THEN: Provide one bounding box for each individual mitochondrion
   - Cover all visible mitochondria that are clearly distinguishable
   - Make boxes tight around each mitochondrion
   - AVOID including other organelles (ER, nuclei, vesicles, Golgi apparatus)
   - Focus on regions with clear mitochondrial morphology only
   - Each box should contain primarily one mitochondrion, not mixed structures
   - Number of boxes should match the number of distinct mitochondria you identify

2. positive_points: [y, x] coordinates INSIDE clear mitochondria (10-20 points)
   - Place points in the center/darkest parts of mitochondria
   - Distribute across different mitochondria
   - Focus on well-defined, clearly visible mitochondria
   - MORE positive points = better segmentation

3. negative_points: [y, x] coordinates in NON-mitochondria regions (15-25 points)
   - Place in nuclei (large dark circular regions)
   - Place in ER (reticular network structures)
   - Place in Golgi apparatus (stacked membrane structures)
   - Place in vesicles and vacuoles
   - Place in background/cytoplasm (areas without mitochondria)
   - Place between closely spaced mitochondria to help separate them
   - MORE negative points = cleaner segmentation, fewer false positives

COORDINATE FORMAT:
- All coordinates normalized to [0, 1000] range
- Based on image dimensions: [0, 1000] maps to [0, width] and [0, height]
- Example: center of image = [500, 500]

OUTPUT FORMAT:
Return ONLY valid JSON (no markdown, no backticks, no extra text):
{{
  "positive_boxes": [[ymin, xmin, ymax, xmax], ...],
  "positive_points": [[y, x], ...],
  "negative_points": [[y, x], ...]
}}

CRITICAL: 
- First assess the image to determine the appropriate number of bounding boxes
- Mitochondria can be small and numerous - ensure each distinct mitochondrion gets its own box
- Provide MORE points (10-20 positive, 15-25 negative) for accurate SAM3 segmentation
- Ensure boxes exclude other organelles to avoid contamination
- Adapt the number of boxes based on mitochondrial density and distribution"""

MITO_DEFAULT_DETECTION ="""- Mitochondria: oval or elongated organelles with characteristic double membrane structure
- Look for dark, bean-shaped or rod-shaped structures
- Often appear as small, discrete organelles scattered throughout the cytoplasm
- May show visible cristae (internal membrane folds) in high-resolution images
"""

MITO_FEEDBACK = """You are an expert electron microscopist evaluating mitochondria segmentation quality.

IMAGE LAYOUT:
- LEFT: Original electron microscopy image showing mitochondrial structures
- RIGHT: Current segmentation result (WHITE = detected as mitochondria, BLACK = background)

TASK: Evaluate segmentation quality by visual comparison of LEFT vs RIGHT images.

EVALUATION CRITERIA:

Carefully examine BOTH images side-by-side. For each criterion below, compare what you see in the LEFT image (ground truth mitochondria) with what is captured in the RIGHT image (segmentation).

1. MORPHOLOGICAL COVERAGE (Weight: 25%)
   Look at the LEFT image: How many distinct mitochondria do you see?
   - Count all mitochondria (round, oval, elongated, tubular)
   - Note size variation (small to large)
   - Identify partial mitochondria at image edges
   
   Now look at RIGHT: What percentage of these mitochondria are captured as white pixels?
   
   Score 0.0-1.0:
   - 1.0: All visible mitochondria captured, regardless of size or shape
   - 0.75: Most mitochondria captured, some small or edge mitochondria missing
   - 0.5: About half the mitochondria captured
   - 0.25: Only a few large/obvious mitochondria captured
   - 0.0: Almost nothing captured
   
   CRITICAL: Be strict. If you see 8 mitochondria in LEFT but only 4 are white in RIGHT, score 0.5, not higher.

2. BOUNDARY ACCURACY (Weight: 25%)
   For each mitochondrion visible in LEFT, examine its outer membrane boundary.
   
   In RIGHT, check if white pixels:
   - Stop exactly at outer membrane (good) or extend into cytoplasm (bad)
   - Cover the full membrane perimeter (good) or have gaps (bad)
   - Follow the true shape (round/oval/tubular) (good) or appear distorted (bad)
   - Include the matrix space (good) or only partial interior coverage (bad)
   
   Score 0.0-1.0:
   - 1.0: Boundaries precisely match outer membrane for all mitochondria
   - 0.75: Most boundaries accurate, slight looseness/tightness in some mitochondria
   - 0.5: Noticeable boundary errors - some too broad, some too narrow, some gaps
   - 0.25: Significant boundary problems throughout
   - 0.0: Boundaries very poor, shapes unrecognizable
   
   CRITICAL: Mitochondria have distinct shapes - boundaries should faithfully reproduce these.

3. SPATIAL ACCURACY (Weight: 20%)
   Compare the spatial extent and completeness in LEFT vs RIGHT.
   
   For each mitochondrion in LEFT:
   - Is the ENTIRE mitochondrion captured (not just a portion)?
   - Are elongated/tubular mitochondria captured along their full length?
   - Are cristae-rich regions included (should be white)?
   - Are partial mitochondria at edges captured proportionally?
   
   Score 0.0-1.0:
   - 1.0: Complete coverage of all mitochondria, full interiors filled
   - 0.75: Most mitochondria fully captured, some partial interior or edge truncation
   - 0.5: Many mitochondria only partially captured (e.g., only portions of interiors)
   - 0.25: Most mitochondria severely truncated, hollow or fragmented
   - 0.0: Only tiny fragments or edge pixels captured

4. CRISTAE AND INTERNAL STRUCTURE (Weight: 15%)
   Look at LEFT: Mitochondrial cristae appear as darker internal folds/lamellae.
   
   Look at RIGHT: Are cristae-containing regions included as white?
   
   For each mitochondrion in LEFT:
   - Should the entire matrix (including cristae regions) be white in RIGHT
   - Cristae are internal structures, not voids to exclude
   - Some mitochondria have dense cristae, others have sparse cristae
   
   Score 0.0-1.0:
   - 1.0: All mitochondria filled completely, cristae regions included
   - 0.75: Most mitochondria well-filled, minor gaps in dense cristae regions
   - 0.5: Moderate gaps in interior, some cristae regions excluded
   - 0.25: Heavy gaps, mitochondria appear hollow or ring-like
   - 0.0: Only outer membranes captured, interiors mostly empty
   
   CRITICAL: Cristae are part of mitochondria - interiors should be solid white, not hollow rings.

5. SPECIFICITY (Weight: 15%)
   Look at areas in LEFT that are clearly NOT mitochondria (ER, Golgi, vesicles, background, nuclei, peroxisomes).
   
   In RIGHT, are these non-mitochondrial regions correctly BLACK?
   
   Check for white pixels in:
   - ER cisternae (flattened, different texture than mitochondria)
   - Golgi stacks
   - Background cytoplasm
   - Nuclear regions
   - Vesicles or other round organelles lacking cristae
   - Peroxisomes (round but different internal texture)
   
   Score 0.0-1.0:
   - 1.0: Only mitochondria are white, all non-mitochondria are black
   - 0.75: Minor false positives (small scattered white pixels in background)
   - 0.5: Moderate false positives (some vesicles or other organelles incorrectly white)
   - 0.25: Major false positives (large non-mitochondrial regions white)
   - 0.0: Severe false positives throughout

SCORING METHODOLOGY:

Step 1: Score each criterion independently (0.0-1.0)
Step 2: Calculate weighted average:
   quality_score = (morphological_coverage × 0.25) + 
                   (boundary_accuracy × 0.25) + 
                   (spatial_accuracy × 0.20) + 
                   (cristae_structure × 0.15) + 
                   (specificity × 0.15)

Step 3: Provide visual guidance for improvement

IMPORTANT CALIBRATION GUIDELINES:
- A score of 0.8+ means segmentation is genuinely excellent (>80% IoU equivalent)
- A score of 0.5-0.7 means partial success with significant room for improvement
- A score below 0.5 means poor performance

OUTPUT FORMAT (JSON only, no markdown, no backticks):
{{
  "quality_score": <float 0.0-1.0, calculated weighted average>,
  "criteria_scores": {{
    "morphological_coverage": <float 0.0-1.0>,
    "boundary_accuracy": <float 0.0-1.0>,
    "spatial_accuracy": <float 0.0-1.0>,
    "cristae_structure": <float 0.0-1.0>,
    "specificity": <float 0.0-1.0>
  }},
  "visual_guidance": "3-5 sentences describing mitochondria detection guidance:

Write balanced guidance that acknowledges what's working AND what needs improvement.

Structure your response based on quality_score:

HIGH QUALITY (0.7-1.0):
Start by describing what the segmentation does well, then suggest refinements.
Example: 'Mitochondria appear as oval/round organelles with visible internal cristae and distinct double membranes. Current detection successfully captures 7 of 8 mitochondria with accurate boundary definition and complete interior filling. Shapes and sizes are well-preserved. To further refine: include the small mitochondrion in the lower left corner and ensure cristae-dense regions in the upper right mitochondrion are fully captured.'

MODERATE QUALITY (0.5-0.7):
Balance strengths and weaknesses, prioritize key improvements.
Example: 'Mitochondria appear as oval organelles with dark cristae folds, ranging from 0.5-1.5 μm diameter. Successfully detecting major mitochondria and capturing general shapes. However, interiors are hollow - cristae regions should be included as part of mitochondrial volume, not excluded. Also missing 2-3 smaller mitochondria. Priority: fill entire mitochondrial interiors including cristae, and detect smaller organelles.'

LOW QUALITY (<0.5):
Briefly note any partial success, then focus on major improvements needed.
Example: 'Mitochondria appear as round/oval organelles (6 total) with characteristic dark internal cristae and clear double membranes. Current approach captures only 2 mitochondria, and only as partial rings. Need to: detect all 6 mitochondria regardless of size, capture complete outer membrane boundaries following true oval/round shapes, fill entire interiors including cristae regions (not hollow), and extend coverage to small and edge-located mitochondria. Cristae are internal structures, not voids.'

ALWAYS include:
1. Brief description of what mitochondria look like in this image (number, shapes, size range)
2. What aspect(s) are working well (if any)
3. What needs improvement (prioritized)
4. Specific actionable guidance using 'Include...', 'Detect...', 'Fill...', 'Avoid...'

Focus on being constructive and specific to THIS image."
}}

Provide honest, balanced assessment that helps improve detection while acknowledging successes.
"""



GOLGI_PROMPT_TEMPLATE = """Analyze this microscopy image to identify Golgi apparatus structures for SAM3 segmentation.

TASK: Provide detailed prompts for SAM3 (Segment Anything Model 3) to accurately segment all Golgi apparatus regions.

WHAT TO DETECT:
{what_to_detect}

PROMPT REQUIREMENTS:
1. positive_boxes: Bounding boxes [ymin, xmin, ymax, xmax] around distinct Golgi regions
   - FIRST: Analyze the image and estimate how many distinct Golgi apparatus regions are visible
   - THEN: Provide one bounding box for each Golgi complex or distinct Golgi region
   - Cover all visible Golgi structures that are clearly distinguishable
   - Make boxes tight around the Golgi apparatus
   - Include the main Golgi stack and closely associated vesicles
   - AVOID including other organelles (mitochondria, ER, nuclei, large vesicles)
   - Focus on regions with clear Golgi morphology (stacked/layered appearance)
   - Each box should contain primarily Golgi apparatus, not mixed structures
   - Number of boxes should match the number of distinct Golgi regions you identify

2. positive_points: [y, x] coordinates INSIDE clear Golgi structures (6-12 points)
   - Place points in the center of Golgi stacks
   - Focus on the densest/darkest parts of the Golgi apparatus
   - Distribute across different Golgi cisternae if multiple stacks are visible
   - Include points in characteristic crescent or curved regions
   - MORE positive points = better segmentation

3. negative_points: [y, x] coordinates in NON-Golgi regions (12-20 points)
   - Place in nuclei (large dark circular regions)
   - Place in ER (reticular network structures)
   - Place in mitochondria (small oval organelles)
   - Place in large vesicles and vacuoles
   - Place in background/cytoplasm (areas without Golgi)
   - Place in regions that might be confused with Golgi (dense ER regions)
   - Place near Golgi boundaries to refine edges
   - MORE negative points = cleaner segmentation, fewer false positives

COORDINATE FORMAT:
- All coordinates normalized to [0, 1000] range
- Based on image dimensions: [0, 1000] maps to [0, width] and [0, height]
- Example: center of image = [500, 500]

OUTPUT FORMAT:
Return ONLY valid JSON (no markdown, no backticks, no extra text):
{{
  "positive_boxes": [[ymin, xmin, ymax, xmax], ...],
  "positive_points": [[y, x], ...],
  "negative_points": [[y, x], ...]
}}

CRITICAL: 
- First assess the image to determine the appropriate number of bounding boxes
- Look for characteristic stacked/layered morphology
- Provide appropriate points (6-12 positive, 12-20 negative) for accurate SAM3 segmentation
- Ensure boxes exclude other organelles to avoid contamination
- Adapt the number of boxes based on Golgi distribution and visibility"""


GOLGI_DEFAULT_DETECTION ="""- Golgi apparatus: stacked, flattened membrane-bound sacs (cisternae) arranged in parallel
- Look for curved or crescent-shaped stacks
- Often appears as compact, layered structures near the nucleus
- May show vesicles budding from the edges"""


GOLGI_FEEDBACK_NEW = """You are an expert in electron microscopy analyzing Golgi apparatus segmentation quality.

IMAGE LAYOUT:
- LEFT: Original electron microscopy image showing Golgi apparatus structures
- RIGHT: Current segmentation (WHITE = detected as Golgi, BLACK = background)

TASK: Evaluate segmentation quality and provide concise Golgi detection guidance.

STEP 1: EVALUATE QUALITY (0.0-1.0 scale)

Score each criterion where:
- 0.8-1.0 = Excellent (minor issues only)
- 0.6-0.8 = Good (captures most structures, some inaccuracies)
- 0.4-0.6 = Fair (partial success, missing some structures)
- 0.2-0.4 = Poor (major gaps or false positives)
- 0.0-0.2 = Very poor (mostly incorrect or missing)

1. STRUCTURE CAPTURE (0.0-1.0): Are the main Golgi stacks and major cisternae detected?
   - 1.0 = All major stacks captured
   - 0.5 = About half of visible Golgi detected
   - 0.0 = No or minimal Golgi detected

2. BOUNDARY QUALITY (0.0-1.0): Do boundaries reasonably follow cisternal edges?
   - 1.0 = Boundaries closely follow membranes
   - 0.5 = Boundaries approximate structure but imprecise
   - 0.0 = Boundaries very inaccurate or merged

3. FALSE POSITIVES (0.0-1.0): Is non-Golgi incorrectly included?
   - 1.0 = Only Golgi segmented
   - 0.5 = Some ER or other organelles included
   - 0.0 = Mostly non-Golgi structures

quality_score = average of 3 criteria (be generous if 2+ structures detected correctly)

STEP 2: PROVIDE CONCISE GOLGI DETECTION GUIDANCE

Based on comparing LEFT to RIGHT, write 3-5 sentences describing what Golgi SHOULD look like for detection.
Focus on what the current segmentation needs to capture better.

OUTPUT FORMAT (JSON only, no markdown, no backticks):
{{
  "quality_score": <float 0.0-1.0>,
  "criteria_scores": {{
    "structure_capture": <float 0.0-1.0>,
    "boundary_quality": <float 0.0-1.0>,
    "false_positives": <float 0.0-1.0>
  }},
  "visual_guidance": "3-5 sentences describing Golgi visual characteristics for detection, based on weaknesses in current segmentation.

Example format (adapt to this specific image):
'Golgi appears as stacked parallel cisternae with darker electron-dense regions. Detect {describe number and arrangement of stacks}. Boundaries should follow the curved membrane edges of each cisterna. {Mention any peripheral vesicles or trans-Golgi network if visible}. Avoid merging with nearby ER tubules or mitochondria.'

Be specific to THIS image's Golgi and current segmentation issues.
Use imperative detection language: 'Detect...', 'Include...', 'Maintain...', 'Avoid...'
NO coordinates or pixel locations."
}}

IMPORTANT: Be calibrated in scoring - if the segmentation captures ANY visible Golgi structures reasonably well, score should be ≥ 0.3. Only score below 0.2 if almost nothing is detected correctly.
"""

GOLGI_FEEDBACK = """You are an expert in electron microscopy analyzing Golgi apparatus segmentation quality.

IMAGE LAYOUT:
- LEFT: Original electron microscopy image showing Golgi apparatus structures
- RIGHT: Current segmentation (WHITE = detected as Golgi, BLACK = background)

TASK: Evaluate segmentation quality and provide concise Golgi detection guidance.

STEP 1: EVALUATE QUALITY BASED ON SPECIFIC CRITERIA

Evaluate by comparing LEFT to RIGHT across six criteria (each 0.0-1.0):

1. MORPHOLOGICAL ACCURACY: Are all Golgi components (cisternal stacks, individual cisternae, associated vesicles) captured?
2. BOUNDARY PRECISION: Do boundaries tightly follow cisternal membrane edges without merging adjacent cisternae?
3. STACK ORGANIZATION: Are parallel cisternae maintained as separate layers with proper intercisternal spacing?
4. COMPLETENESS: What percentage of visible Golgi structures in LEFT (including peripheral regions and vesicles) is captured in RIGHT?
5. SPECIFICITY: Is ONLY Golgi segmented (no ER, mitochondria, endosomes, or other organelles)?
6. STRUCTURAL FIDELITY: Is the characteristic stacked, polarized architecture (cis-to-trans) preserved?

quality_score = average of 6 criteria

STEP 2: PROVIDE CONCISE GOLGI DETECTION GUIDANCE

Based on comparing LEFT to RIGHT, write 3-5 sentences describing what Golgi SHOULD look like for detection.
Focus on what the current segmentation needs to capture better.

OUTPUT FORMAT (JSON only, no markdown, no backticks):
{{
  "quality_score": <float 0.0-1.0>,
  "criteria_scores": {{
    "morphological_accuracy": <float 0.0-1.0>,
    "boundary_precision": <float 0.0-1.0>,
    "stack_organization": <float 0.0-1.0>,
    "completeness": <float 0.0-1.0>,
    "specificity": <float 0.0-1.0>,
    "structural_fidelity": <float 0.0-1.0>
  }},
  "visual_guidance": "3-5 sentences describing Golgi visual characteristics for detection, based on weaknesses in current segmentation.

Example format (adapt to this specific image):
'Golgi appears as [describe stacks and cisternae]. Detect [number] separate cisternae with [boundary characteristics]. Maintain [spacing between layers]. Include [peripheral structures like vesicles]. Avoid [incorrectly merged or misclassified structures].'

Be specific to THIS image's Golgi and current segmentation issues.
Use imperative detection language: 'Detect...', 'Include...', 'Maintain...', 'Avoid...'
NO coordinates or pixel locations."
}}

Keep it concise - focus on the most critical detection guidance based on current results.
"""


# GOLGI_FEEDBACK = """You are an expert electron microscopist evaluating Golgi apparatus segmentation quality.

# IMAGE LAYOUT:
# - LEFT: Original electron microscopy image showing Golgi apparatus structures
# - RIGHT: Current segmentation result (WHITE = detected as Golgi, BLACK = background)

# TASK: Evaluate segmentation quality by visual comparison of LEFT vs RIGHT images.

# EVALUATION CRITERIA:

# Carefully examine BOTH images side-by-side. For each criterion below, compare what you see in the LEFT image (ground truth Golgi) with what is captured in the RIGHT image (segmentation).

# 1. MORPHOLOGICAL COVERAGE (Weight: 25%)
#    Look at the LEFT image: How many Golgi structures do you see?
#    - Count distinct Golgi stacks (multiple parallel cisternae)
#    - Count individual cisternae within each stack
#    - Identify trans-Golgi network (TGN) regions
#    - Count associated vesicles in cis/trans faces
   
#    Now look at RIGHT: What percentage of these structures are captured as white pixels?
   
#    Score 0.0-1.0:
#    - 1.0: All visible Golgi stacks and cisternae captured
#    - 0.75: Most major stacks captured, some peripheral cisternae or vesicles missing
#    - 0.5: About half the stacks/cisternae captured
#    - 0.25: Only a few stacks captured
#    - 0.0: Almost nothing captured
   
#    CRITICAL: Be strict. If you see a stack with 5 cisternae in LEFT but only 2-3 are white in RIGHT, score ~0.5, not higher.

# 2. BOUNDARY ACCURACY (Weight: 25%)
#    For each Golgi cisterna visible in LEFT, examine its boundaries.
   
#    In RIGHT, check if white pixels:
#    - Stop exactly at cisternal membrane edges (good) or extend into intercisternal space (bad)
#    - Cover the full cisternal width (good) or only partial coverage (bad)
#    - Maintain the characteristic flattened, curved morphology (good) or appear bloated/irregular (bad)
#    - Preserve the parallel stacking arrangement (good) or merge adjacent cisternae (bad)
   
#    Score 0.0-1.0:
#    - 1.0: Boundaries precisely match cisternal edges, intercisternal spaces preserved
#    - 0.75: Most boundaries accurate, slight merging or gaps in some cisternae
#    - 0.5: Noticeable boundary errors - cisternae merged or poorly defined
#    - 0.25: Significant boundary problems, stacks not distinguishable
#    - 0.0: Boundaries very poor, blob-like regions
   
#    CRITICAL: Golgi cisternae should remain distinct, not merged into solid blocks.

# 3. SPATIAL ACCURACY (Weight: 20%)
#    Compare the spatial extent and organization in LEFT vs RIGHT.
   
#    For each Golgi stack in LEFT:
#    - Is the FULL LATERAL EXTENT captured (edge to edge of each cisterna)?
#    - Are the curved/fenestrated edges of cisternae included?
#    - Is the polarized organization (cis to trans) preserved?
#    - Are peripheral vesicles at cis/trans faces captured?
   
#    Score 0.0-1.0:
#    - 1.0: Complete spatial extent of all stacks, cisternae, and vesicles captured
#    - 0.75: Most stacks captured fully, some edge regions or vesicles truncated
#    - 0.5: Many cisternae only partially captured (e.g., only central portions)
#    - 0.25: Most cisternae severely truncated or vesicles omitted
#    - 0.0: Only tiny fragments captured

# 4. STACK ORGANIZATION PRESERVATION (Weight: 15%)
#    Look at LEFT: Golgi cisternae are organized in parallel stacks with characteristic spacing.
   
#    Look at RIGHT: Is this organized architecture preserved?
   
#    For each Golgi stack in LEFT:
#    - Are cisternae maintained as separate parallel layers?
#    - Is the intercisternal space (~15-20nm) preserved as BLACK?
#    - Are cisternae ordered from cis to trans?
#    - Are associated vesicles positioned correctly relative to stack?
   
#    Score 0.0-1.0:
#    - 1.0: All stacks maintain proper parallel organization and spacing
#    - 0.75: Minor issues (1-2 cisternae slightly merged, mostly organized)
#    - 0.5: Moderate organization loss (several cisternae merged, spacing unclear)
#    - 0.25: Heavy organization loss (stacks appear as solid masses)
#    - 0.0: Complete loss of stack organization

# 5. SPECIFICITY (Weight: 15%)
#    Look at areas in LEFT that are clearly NOT Golgi (ER, mitochondria, background, nuclei, other vesicles).
   
#    In RIGHT, are these non-Golgi regions correctly BLACK?
   
#    Check for white pixels in:
#    - ER cisternae/tubules (should be distinguishable from Golgi)
#    - Mitochondria with cristae
#    - Endosomes or lysosomes
#    - Background cytoplasm
#    - Nuclear envelope
   
#    Score 0.0-1.0:
#    - 1.0: Only Golgi is white, all non-Golgi is black
#    - 0.75: Minor false positives (small scattered white pixels, few misclassified vesicles)
#    - 0.5: Moderate false positives (some ER or other organelles incorrectly white)
#    - 0.25: Major false positives (large non-Golgi regions white)
#    - 0.0: Severe false positives throughout

# SCORING METHODOLOGY:

# Step 1: Score each criterion independently (0.0-1.0)
# Step 2: Calculate weighted average:
#    quality_score = (morphological_coverage × 0.25) + 
#                    (boundary_accuracy × 0.25) + 
#                    (spatial_accuracy × 0.20) + 
#                    (stack_organization × 0.15) + 
#                    (specificity × 0.15)

# Step 3: Provide visual guidance for improvement

# IMPORTANT CALIBRATION GUIDELINES:
# - A score of 0.8+ means segmentation is genuinely excellent (>80% IoU equivalent)
# - A score of 0.5-0.7 means partial success with significant room for improvement
# - A score below 0.5 means poor performance

# OUTPUT FORMAT (JSON only, no markdown, no backticks):
# {{
#   "quality_score": <float 0.0-1.0, calculated weighted average>,
#   "criteria_scores": {{
#     "morphological_coverage": <float 0.0-1.0>,
#     "boundary_accuracy": <float 0.0-1.0>,
#     "spatial_accuracy": <float 0.0-1.0>,
#     "stack_organization": <float 0.0-1.0>,
#     "specificity": <float 0.0-1.0>
#   }},
#   "visual_guidance": "3-5 sentences describing Golgi detection guidance:

# Write balanced guidance that acknowledges what's working AND what needs improvement.

# Structure your response based on quality_score:

# HIGH QUALITY (0.7-1.0):
# Start by describing what the segmentation does well, then suggest refinements.
# Example: 'Golgi appears as parallel stacks of 4-6 curved cisternae with characteristic fenestrated edges and associated vesicles. Current detection successfully captures the main cisternal stack with good boundary definition and preserved intercisternal spacing. Stack organization maintained from cis to trans faces. To further refine: include peripheral vesicles at trans face and extend coverage to fenestrated cisternal edges.'

# MODERATE QUALITY (0.5-0.7):
# Balance strengths and weaknesses, prioritize key improvements.
# Example: 'Golgi appears as organized stacks of flattened cisternae with surrounding transport vesicles. Successfully detecting major cisternae in central stack regions. However, boundaries merge adjacent cisternae - should preserve intercisternal spacing as distinct layers. Also missing peripheral vesicles and edge regions. Priority: maintain separation between individual cisternae and include complete lateral extent.'

# LOW QUALITY (<0.5):
# Briefly note any partial success, then focus on major improvements needed.
# Example: 'Golgi appears as 5-7 parallel curved cisternae stacked with regular spacing. Current approach captures only 1-2 central cisternae as merged blob. Need to: detect all individual cisternae as separate parallel layers, maintain intercisternal spacing, include full lateral extent with fenestrated edges, capture associated vesicles, and preserve cis-to-trans organization. Avoid merging cisternae into solid masses.'

# ALWAYS include:
# 1. Brief description of what Golgi looks like in this image (number of cisternae, organization)
# 2. What aspect(s) are working well (if any)
# 3. What needs improvement (prioritized)
# 4. Specific actionable guidance using 'Include...', 'Detect...', 'Maintain...', 'Avoid...'

# Focus on being constructive and specific to THIS image."
# }}

# Provide honest, balanced assessment that helps improve detection while acknowledging successes.
# """
