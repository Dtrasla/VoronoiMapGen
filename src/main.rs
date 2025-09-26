
use rand::{rngs::ThreadRng, Rng, seq::SliceRandom};
use voronoice::{BoundingBox, ClipBehavior, Point, Voronoi, VoronoiBuilder, VoronoiCell};
use delaunator::{EMPTY, next_halfedge};
use std::{collections::{HashSet, VecDeque}, path};

const CANVAS_SIZE: f64 = 2200.;
const CANVAS_MARGIN: f64 = 0.;
const POINT_SIZE: usize = 2;
const CIRCUMCENTER_CIRCLE_COLOR: &str = "black";
const SITE_COLOR: &str = "black";
const CIRCUMCENTER_COLOR: &str = "red";
const LINE_WIDTH: usize = 1;
const VORONOI_EDGE_COLOR: &str = "blue";
const TRIANGULATION_HULL_COLOR: &str = "green";
const TRIANGULATION_LINE_COLOR: &str = "grey";
const JITTER_RANGE_VALUE: f64 = 5.;
const SAND_COLOR: &str = "peachpuff";
const OCEAN_COLOR: &str = "steelblue";
const FRESH_WATER_COLOR: &str = "lightskyblue";
const RIVER_COLOR: &str = "lightseagreen";
const MAINLAND_COLOR: &str = "olivedrab";
const GAMMA : f32 = 0.68; // Probability of continuing land growth
const RIVER_GAMMA: f32 = 0.85;
const RIVER_EXPANSION: f32 = 0.5;
const MAX_RIVERS: usize = 5;
const MAX_STEPS: usize = 1000;


const W_DOWN: f64 = 2.0;
const W_SEA: f64  = 1.0;
const W_CURVE: f64 = 0.6;
const W_CONF: f64 = 3.0;   // prefer merging
const NOISE_JITTER: f64 = 0.1;
const ALPHA_ELEV: f64 = 1.0; // dist-to-coast weight
const BETA_NOISE: f64 = 0.35; // noise weight

fn main() {
    let n = 16384;
    let sites = generate_points(n);

    //let sites: Vec<_> = points.iter().map(|(x, y)| Point { x: *x, y: *y }).collect();

    let bbox = BoundingBox::new(Point { x: (0.0), y: (0.0) }, 2100.0, 2100.0);
    let transform = build_transformation(&sites);

    let sites: Vec<Point> = sites.iter().map(|&[x, y]| Point { x, y }).collect();

    let diagram = VoronoiBuilder::default()
        .set_sites(sites.clone())
        .set_bounding_box(bbox)
        .set_lloyd_relaxation_iterations(1)
        .build()
        .unwrap();

    
    let n = diagram.cells().len();
    let mut land: Vec<usize> = Vec::with_capacity(n); 
    let mut salt_water: Vec<usize> = Vec::with_capacity(n);
    let mut fresh_water: Vec<usize> = Vec::with_capacity(n);

    islands(&diagram, 8, &mut land, &mut salt_water, &mut fresh_water);
    let rivers = generate_rivers(&diagram, &land, &fresh_water);
    let coast = separate_coast_and_inland(&diagram, &mut land, &salt_water);

    let svg = voronoi_to_svg(&diagram, &transform, &coast, &land,&salt_water, &fresh_water, &rivers);

    std::fs::write("voronoi.svg", svg).expect("write svg");
    println!("Wrote voronoi.svg");
}


fn generate_points(n: usize) -> Vec<[f64; 2]> {

    let mut rng = rand::rng();
    let x_range = rand::distr::Uniform::new(-1000.0, 1000.0)
        .expect("Failed to create x_range");
    let y_range = rand::distr::Uniform::new(-1000.0, 1000.0)
        .expect("Failed to create y_range");

    let mut rng = rand::rng();
    let mut points: Vec<(f64, f64)> = Vec::with_capacity(n);

    for _ in 0..n {
        let x: f64 = rng.random_range(-1000.0..=1000.0);
        let y: f64 = rng.random_range(-1000.0..=1000.0);
        points.push((x, y));
    }

    (0..n).map(move |_| [rng.sample(x_range), rng.sample(y_range)])
        .collect::<Vec<_>>()
}


/*TERRAIN GENERATION */

fn islands(voronoi: &Voronoi, k: usize, land_cells: &mut Vec<usize>, salt_water: &mut Vec<usize>, fresh_water: &mut Vec<usize>) {
    let mut rng = rand::rng();
    let n = voronoi.cells().len();

    // Generate blobs of land randomly using recursion and gamma as chance of land expanding
    for _ in 0..k {
        let cell_index = rng.random_range(0..n);
        let cell = voronoi.cell(cell_index);
        //land_max_depth(voronoi, &cell, &mut cells, cell_index, &mut rng, 20);
        land(voronoi, &cell, land_cells, cell_index, &mut rng);
    }

    // Generate water cells based on islands, using same recursive approach but lower probability
    generate_water_from_empty_cells(voronoi, fresh_water, salt_water, land_cells);

}

// From mainland points, recursively walk to neighbors, depends on GAMMA probability to expand land
fn land (diagram: &Voronoi, cell: &VoronoiCell, cells: &mut Vec<usize>, cellpos: usize, rng: &mut ThreadRng) {
    cells.push(cellpos);

    for neighbor in cell.iter_neighbors() {    
        if (rng.random_range(0.0..1.0) > GAMMA) && !cells.contains(&neighbor) {
            land(diagram, &diagram.cell(neighbor), cells, neighbor, rng);
        }
    }
}

// Useful for lower GAMMA values to avoid enormous landmasses
fn land_max_depth (diagram: &Voronoi, cell: &VoronoiCell, cells: &mut Vec<usize>, cellpos: usize, rng: &mut ThreadRng, depth: usize) {
    cells.push(cellpos);

    if depth == 0 {
        return; 
    }
    for neighbor in cell.iter_neighbors() {    
        let tuple = diagram.cell(neighbor).site_position().clone();

        if (rng.random_range(0.0..1.0) > GAMMA) && !cells.contains(&neighbor) {
            land_max_depth(diagram, &diagram.cell(neighbor), cells, neighbor, rng, depth - 1);
        }
    }
}

fn separate_coast_and_inland(diagram: &Voronoi, land_cells: &mut Vec<usize>, salt_water: &Vec<usize>) -> Vec<usize> {
    let mut coast = Vec::with_capacity(land_cells.len());
    let mut values_to_remove = Vec::new();
    for cell in &land_cells.clone() {
        for neighbor in diagram.cell(*cell).iter_neighbors() {
            if (salt_water.contains(&neighbor)) {
                coast.push(*cell);
                values_to_remove.push(*cell);
                continue;
            }
        }
    }

    land_cells.retain(|x| !values_to_remove.contains(x));

    coast
}

pub fn generate_water_from_empty_cells(
    diagram: &Voronoi,
    fresh_water: &mut Vec<usize>,
    salt_water: &mut Vec<usize>,
    island_cells: &mut Vec<usize>,
) {
    let island: HashSet<usize> = island_cells.iter().copied().collect();

    // Cells we’ve already labeled as water (fresh or salt) to avoid reprocessing
    let mut labeled: HashSet<usize> =
        fresh_water.iter().chain(salt_water.iter()).copied().collect();

    // Explore only empty neighbors of land. Each un-labeled empty region gets a single flood fill
    for &land_idx in &island {
        let land_cell = diagram.cell(land_idx);
        for neighbor in land_cell.iter_neighbors() {
            if island.contains(&neighbor) || labeled.contains(&neighbor) {
                continue;
            }
            // neighbor is an empty cell we haven't labeled yet → flood-fill its component
            let (component, touches_hull) =
                flood_fill_empty_component(diagram, neighbor, &island, &labeled);

            // Mark them as labeled (so we never touch these again).
            labeled.extend(component.iter().copied());

            // Push into the appropriate bucket
            if touches_hull {
                salt_water.extend(component.iter().copied());
            } else {
                fresh_water.extend(component.iter().copied());
            }
        }
    }

    dedup_in_place(fresh_water);
    dedup_in_place(salt_water);
}

fn flood_fill_empty_component(
    diagram: &Voronoi,
    start: usize,
    island: &HashSet<usize>,
    labeled: &HashSet<usize>,
) -> (Vec<usize>, bool) {
    let mut q = VecDeque::new();
    let mut seen: HashSet<usize> = HashSet::new();
    let mut component: Vec<usize> = Vec::new();
    let mut touches_hull = false;

    q.push_back(start);
    seen.insert(start);

    while let Some(idx) = q.pop_front() {
        let cell = diagram.cell(idx);
        if cell.is_on_hull() {
            touches_hull = true;
        }

        component.push(idx);

        // Explore neighbors that are also empty (not land) and not yet labeled/seen
        for nb in cell.iter_neighbors() {
            if island.contains(&nb) || labeled.contains(&nb) || seen.contains(&nb) {
                continue;
            }
            seen.insert(nb);
            q.push_back(nb);
        }
    }

    (component, touches_hull)
}

fn dedup_in_place(v: &mut Vec<usize>) {
    let mut seen = HashSet::new();
    v.retain(|x| seen.insert(*x));
}


fn land_set(land_cells: &Vec<usize>) -> HashSet<usize> {
    land_cells.iter().copied().collect()
}

fn dist_to_coast(diagram: &Voronoi, land: &HashSet<usize>) -> Vec<i32> {
    // BFS in land graph starting at hull land cells
    let n = diagram.cells().len();
    let mut dist = vec![i32::MAX; n];
    let mut q = VecDeque::new();

    for &ci in land {
        if diagram.cell(ci).is_on_hull() {
            dist[ci] = 0;
            q.push_back(ci);
        }
    }
    while let Some(u) = q.pop_front() {
        let du = dist[u];
        for v in diagram.cell(u).iter_neighbors() {
            if !land.contains(&v) { continue; }
            if dist[v] == i32::MAX {
                dist[v] = du + 1;
                q.push_back(v);
            }
        }
    }
    dist
}

// Minimal fBm-ish noise using site coords (replace with `noise` crate if you prefer)
fn hash_noise(p: &Point) -> f64 {
    let x = (p.x * 127.1 + p.y * 311.7).sin();
    let y = (p.x * 269.5 + p.y * 183.3).cos();
    (x + y) as f64 * 0.5
}

fn build_elevation(diagram: &Voronoi, land: &HashSet<usize>, dist: &Vec<i32>) -> Vec<f64> {
    let mut elev = vec![0.0; diagram.cells().len()];
    let maxd = dist.iter().copied().filter(|d| *d < i32::MAX).max().unwrap_or(1) as f64;
    for &ci in land {
        let d = dist[ci] as f64 / maxd;                 // 0 (coast) .. 1 (interior)
        let n = hash_noise(diagram.cell(ci).site_position());
        elev[ci] = ALPHA_ELEV * d + BETA_NOISE * n;    // bumpy downhill toward coast
    }
    elev
}

// ---- River generation ----

pub fn generate_rivers(
    diagram: &Voronoi,
    land_cells: &Vec<usize>,
    fresh_water: &Vec<usize>,   // lakes to seed
) -> Vec<Vec<Point>> {
    let mut rng = rand::rng();
    let land = land_set(land_cells);
    let dsea = dist_to_coast(diagram, &land);
    let elev = build_elevation(diagram, &land, &dsea);
    let mut is_river = vec![false; diagram.cells().len()];
    let mut rivers: Vec<Vec<Point>> = Vec::new();

    // Seeds: any lake neighbor that is land and not already a river
    for &lake_idx in fresh_water {
        for nb in diagram.cell(lake_idx).iter_neighbors() {
            if !land.contains(&nb) || is_river[nb] { continue; }

            if rng.random_range(0.0..1.0) > 0.45 {
                let mut path_idx = walk_river(diagram, nb, &land, &elev, &dsea, &mut is_river, &mut rng);
                if path_idx.len() >= 3 {
                    let mut poly: Vec<Point> = path_idx
                        .drain(..)
                        .map(|ci| diagram.cell(ci).site_position().clone())
                        .collect();
                    poly = chaikin_smooth(poly, 2);
                    rivers.push(poly);
                }
            }
        }
    }
    rivers
}

fn walk_river(
    diagram: &Voronoi,
    start: usize,
    land: &HashSet<usize>,
    elev: &Vec<f64>,
    dsea: &Vec<i32>,
    is_river: &mut Vec<bool>,
    rng: &mut ThreadRng,
) -> Vec<usize> {
    let mut path = Vec::new();
    let mut visited = HashSet::new();
    let mut cur = start;
    let mut prev_dir: Option<(f64,f64)> = None;
    let mut steps = 0usize;

    while steps < MAX_STEPS {
        steps += 1;
        let cell = diagram.cell(cur);
        path.push(cur);
        is_river[cur] = true;
        visited.insert(cur);

        if cell.is_on_hull() { break; } // reached the sea

        // Candidate next land neighbors
        let mut best = None;
        let e_cur = elev[cur];
        let d_cur = dsea[cur];

        for nb in cell.iter_neighbors() {
            if !land.contains(&nb) || visited.contains(&nb) { continue; }

            // prefer merging into an existing river
            let conf = if is_river[nb] { 1.0 } else { 0.0 };

            // downhill & to-sea terms
            let e_gain = (e_cur - elev[nb]).max(0.0);
            let d_gain = (d_cur - dsea[nb]) as f64;

            // curvature term
            let dir = {
                let a = cell.site_position();
                
                let b = diagram.cell(nb).site_position().clone();
                let vx = (b.x - a.x) as f64;
                let vy = (b.y - a.y) as f64;
                let len = (vx*vx + vy*vy).sqrt().max(1e-6);
                (vx/len, vy/len)
            };
            let curve = if let Some(pd) = prev_dir {
                pd.0 * dir.0 + pd.1 * dir.1   // cosine
            } else { 0.0 };

            let jitter = (rng.random_range(-1.0..1.0) as f64) * NOISE_JITTER;

            let score = W_DOWN*e_gain + W_SEA*d_gain + W_CURVE*curve + W_CONF*conf + jitter;

            if best.map_or(true, |(_, s, _)| score > s) {
                best = Some((nb, score, dir));
            }
        }

        match best {
            None => break, // dead end
            Some((next_idx, _, dir)) => {
                // If next is already a river, *snap* and stop (confluence).
                if is_river[next_idx] {
                    path.push(next_idx);
                    break;
                }
                prev_dir = Some(dir);
                cur = next_idx;
            }
        }
    }

    path
}

// ---- Light smoothing for nicer meanders ----
fn chaikin_smooth(mut poly: Vec<Point>, iterations: usize) -> Vec<Point> {
    for _ in 0..iterations {
        if poly.len() < 3 { break; }
        let mut out = Vec::with_capacity(poly.len()*2);
        out.push(poly[0].clone()); // preserve ends
        for i in 0..poly.len()-1 {
            let p = &poly[i];
            let q = &poly[i+1];
            let p1 = Point { x: 0.75*p.x + 0.25*q.x, y: 0.75*p.y + 0.25*q.y };
            let q1 = Point { x: 0.25*p.x + 0.75*q.x, y: 0.25*p.y + 0.75*q.y };
            out.push(p1);
            out.push(q1);
        }
        out.push(poly[poly.len()-1].clone());
        poly = out;
    }
    poly
}



// RENDERING

pub fn voronoi_to_svg(voronoi: &Voronoi, transform: &Transform, coast: &[usize], mainland: &[usize],salt_water: &[usize], fresh_water: &[usize], rivers: &[Vec<Point>]) -> String {
    let bounding_box_top_left = transform.transform(&Point { x: voronoi.bounding_box().left(), y: voronoi.bounding_box().top() });
    let bounding_box_side = transform.transform(voronoi.bounding_box().bottom_left()).y - bounding_box_top_left.y;

    let contents = format!(
            r#"
    <svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">
    <rect width="100%" height="100%" fill="white" />
    <rect x="{bb_x}" y="{bb_y}" width="{bb_width}" height="{bb_height}" style="fill-opacity:0;stroke-opacity:0.25;stroke-width:3;stroke:rgb(0,0,0)" />
        {sites}
        {circumcenters}
        {voronoi_edges}
        {triangles}
        {circumcenter_circles}
        {coast}
        {mainland}
        {ocean}
        {fresh}
        {rivers}
    </svg>"#,
            width = CANVAS_SIZE,
            height = CANVAS_SIZE,
            bb_x = bounding_box_top_left.x,
            bb_y = bounding_box_top_left.y,
            bb_width = bounding_box_side,
            bb_height = bounding_box_side,
            sites = "",//render_point(&transform, voronoi.sites(), SITE_COLOR, false, false),
            circumcenters = "",//render_point(&transform, voronoi.vertices(), CIRCUMCENTER_COLOR, true, true),
            voronoi_edges =  "",//render_voronoi_edges(&transform, &voronoi),
            triangles = "", //render_triangles(&transform, &voronoi, false, true),
            circumcenter_circles = "",//render_circumcenters(&transform, &voronoi),
            coast = island_group_svg(voronoi, &transform, coast, SAND_COLOR, 1.0),
            mainland = island_group_svg(voronoi, &transform, mainland, MAINLAND_COLOR, 1.0),
            ocean = island_group_svg(voronoi, &transform, salt_water, OCEAN_COLOR, 1.0),
            fresh = island_group_svg(voronoi, &transform, fresh_water, FRESH_WATER_COLOR, 1.0),
            rivers = rivers_svg_from_points_cased(&transform, rivers, FRESH_WATER_COLOR, "white", 3.0, 5.0, 1.0, 0.3)
        );

        contents
}


fn render_triangles(transform: &Transform, voronoi: &Voronoi, labels: bool, edges: bool) -> String {
    let triangulation = voronoi.triangulation();
    let points = voronoi.sites();

    (0..triangulation.triangles.len()).fold(String::new(), |acc, e| {
        if e > triangulation.halfedges[e] || triangulation.halfedges[e] == EMPTY {
            let start = transform.transform(&points[triangulation.triangles[e]]);
            let end = transform.transform(&points[triangulation.triangles[next_halfedge(e)]]);
            let mid = Point { x: (start.x + end.x) / 2.0, y: (start.y + end.y) / 2.0 };
            let (color, label) = if triangulation.halfedges[e] == EMPTY {
                (TRIANGULATION_HULL_COLOR, format!("{e}"))
            } else {
                (TRIANGULATION_LINE_COLOR, format!("{e} ({})", triangulation.halfedges[e]))
            };

            let acc = if edges {
                acc + &format!(r#"<line id="dedge_{id}" stroke-dasharray="10,10" x1="{x0}" y1="{y0}" x2="{x1}" y2="{y1}" style="stroke:{color};stroke-width:{width}" />"#,
                id = e,
                x0 = start.x,
                y0 = start.y,
                x1=end.x,
                y1=end.y,
                width = LINE_WIDTH,
                color = color)
            } else {
                acc
            };

            if labels {
                acc + &format!(r#"<text x="{x}" y="{y}" style="stroke:{color};">{label}</text>"#, x = mid.x, y = mid.y)
            } else {
                acc
            }
        } else {
            acc
        }
    })
}

fn render_point(transform: &Transform, points: &[Point], color: &str, jitter: bool, labels: bool) -> String {
    let mut rng = rand::rng();
    let jitter_range = rand::distr::Uniform::new(-JITTER_RANGE_VALUE, JITTER_RANGE_VALUE)
    .expect("Failed to create jitter range");

     

    points
        .iter()
        .enumerate()
        .fold(String::new(), |acc, (i, p)| {
            let p = transform.transform(p);
            let (x, y) = if jitter {
                (p.x + rng.sample(jitter_range), p.y + rng.sample(jitter_range))
            } else {
                 (p.x, p.y)
            };

            acc + &format!(
                r#"<circle id="pt_{pi}" cx="{x}" cy="{y}" r="{size}" fill="{color}"/>"#,
                pi = i,
                size = POINT_SIZE,
                color = color
            ) + &if labels { format!(r#"<text x="{x}" y="{y}" style="stroke:{color};">{text}</text>"#, text = i) } else { "".to_string() }
        })
}

fn render_circumcenters(transform: &Transform, voronoi: &Voronoi) -> String {
    voronoi.vertices().iter().enumerate().fold(String::new(), |acc, (triangle, circumcenter)| {
        if triangle < voronoi.triangulation().triangles.len() / 3  {
            let circumcenter = transform.transform(circumcenter);
            let point_on_circle = transform.transform(&voronoi.sites()[voronoi.triangulation().triangles[triangle * 3]]);
            let radius = ((point_on_circle.x - circumcenter.x).powi(2) + (point_on_circle.y - circumcenter.y).powi(2)).sqrt();

            acc + &format!(
                r#"<circle id="ct_{pi}" cx="{x}" cy="{y}" r="{radius}" fill="none" stroke="{color}" stroke-opacity="0.25" />"#,
                pi = triangle,
                x = circumcenter.x,
                y = circumcenter.y,
                color = CIRCUMCENTER_CIRCLE_COLOR
            )
        } else {
            acc
        }
    })
}

fn render_voronoi_edges(transform: &Transform, voronoi: &Voronoi) -> String {
    let mut buffer = String::new();

    for cell in voronoi.iter_cells() {

        let render = |(start, end)| {
            let start = transform.transform(start);
            let end = transform.transform(end);

            buffer += &format!(r#"<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y1}" style="stroke:{color};stroke-width:{width}" />"#,
                x0 = start.x,
                y0 = start.y,
                x1 = end.x,
                y1 = end.y,
                width = LINE_WIDTH,
                color = VORONOI_EDGE_COLOR);
        };

        if let Some(first) = cell.iter_vertices().next() {
                cell.iter_vertices().zip(cell.iter_vertices().skip(1).chain(std::iter::once(first))).for_each(render);
        }
    }

    buffer
}


fn island_group_svg(diagram: &Voronoi, transform: &Transform,cell_indices: &[usize], fill: &str, opacity: f32) -> String {
    let mut s = format!(r#"<g fill="{fill}" stroke="none" opacity="{opacity}">"#);
    for &ci in cell_indices {
        let cell = diagram.cell(ci);
        let mut pts = Vec::new();
        for v in cell.iter_vertices() {
            let v = transform.transform(v);
            pts.push(format!("{:.2},{:.2}", v.x, v.y));
        }
        if pts.len() >= 3 {
            s.push_str(&format!(r#"<polygon points="{}"/>"#, pts.join(" ")));
        }
    }
    s.push_str("</g>");
    s
}

pub fn rivers_svg_from_points(
    transform: &Transform,
    rivers: &[Vec<Point>],
    stroke: &str,
    width: f64,
    opacity: f32,
) -> String {
    let mut s = format!(
        r#"<g fill="none" stroke="{stroke}" stroke-width="{width}" stroke-linecap="round" stroke-linejoin="round" opacity="{opacity}">"#
    );

    for poly in rivers {
        if poly.len() < 2 { continue; }
        let pts: Vec<String> = poly.iter()
            .map(|p| {
                let t = transform.transform(p);
                format!("{:.2},{:.2}", t.x, t.y)
            })
            .collect();
        s.push_str(&format!(r#"<polyline points="{}"/>"#, pts.join(" ")));
    }

    s.push_str("</g>");
    s
}

/// Same as above, but draws a thicker "bank" under-stroke for contrast.
pub fn rivers_svg_from_points_cased(
    transform: &Transform,
    rivers: &[Vec<Point>],
    stroke_main: &str,
    stroke_casing: &str,
    width_main: f64,
    width_casing: f64,
    opacity_main: f32,
    opacity_casing: f32,
) -> String {
    // Draw casing first (under), then the main line (over).
    let under = rivers_svg_from_points(transform, rivers, stroke_casing, width_casing, opacity_casing);
    let over  = rivers_svg_from_points(transform, rivers, stroke_main,   width_main,   opacity_main);
    format!("{under}{over}")
}



pub struct Transform {
    scale: f64,
    center: Point,
    offset: Point,
    farthest_distance: f64,
    rotation: f64
}


impl Transform {
    pub fn transform<T : std::borrow::Borrow<Point>>(&self, p: T) -> Point {
        let p = p.borrow();
        Point {
            x: self.scale * (p.x *  self.rotation.cos() - p.y * self.rotation.sin()) + self.offset.x,
            y: self.scale * (p.y *  self.rotation.cos() + p.x * self.rotation.sin()) + self.offset.y,
        }
    }
}


pub fn build_transformation(points: &[[f64;2]]) -> Transform {
    let mut center = points.iter().fold([0., 0.], |acc, p| [acc[0] + p[0], acc[1] + p[1]]);
    center[0] /= points.len() as f64;
    center[1] /= points.len() as f64;

    let farthest_distance = points
        .iter()
        .map(|p| {
            let (x, y) = (center[0] - p[0], center[1] - p[1]);
            x * x + y * y
        })
        .reduce(f64::max)
        .unwrap()
        .sqrt();

    let scale = 1.0 * CANVAS_SIZE / (farthest_distance * 2.0);
    let offset = Point {
        x: (CANVAS_SIZE / 2.0) - (scale * center[0]) - 0.,
        y: (CANVAS_SIZE / 2.0) - (scale * center[1]) - 0.,
    };

    let center = Point { x: center[0], y: center[1] };

    Transform {
        center,
        scale,
        offset,
        farthest_distance,
        rotation: 0.0_f64.to_radians(),
    }
}
