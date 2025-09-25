
use rand::{rngs::ThreadRng, Rng};
use voronoice::{BoundingBox, ClipBehavior, Point, Voronoi, VoronoiBuilder, VoronoiCell};
use delaunator::{EMPTY, next_halfedge};

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
const SAND_COLOR: &str = "SandyBrown";
const GAMMA : f32 = 0.7; // Probability of continuing land growth


fn main() {
    let n = 4096;//16384;
    let sites = generate_points(n);

    //let sites: Vec<_> = points.iter().map(|(x, y)| Point { x: *x, y: *y }).collect();

    let bbox = BoundingBox::new(Point { x: (0.0), y: (0.0) }, 2100.0, 2100.0);
    println!("bbox: {:?}", bbox.center());
    let transform = build_transformation(&sites);

    let sites: Vec<Point> = sites.iter().map(|&[x, y]| Point { x, y }).collect();
    let cell_indices = pick_island_cells(&sites, n/6 , bbox.width() - 400.0, bbox.height() - 400.0);


    let diagram = VoronoiBuilder::default()
        .set_sites(sites.clone())
        .set_bounding_box(bbox)
        .set_lloyd_relaxation_iterations(1)
        .build()
        .unwrap();

    

    let cells = islands(&diagram, 4);
    // Pick some cells to form an island
    //println!("Iterator: {:?}", diagram.cell(40).iter_neighbors().map());
    //diagram.cell(0).site();
    
    let svg = voronoi_to_svg(&diagram, &transform, &cells);

    for vert in diagram.cell(0).iter_vertices() {
        println!("{:?} {:?}", vert.x, vert.y);
    }

    std::fs::write("voronoi.svg", svg).expect("write svg");
    println!("Wrote voronoi.svg");
    //println!("{:?}", diagram);
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

    //points

    (0..n).map(move |_| [rng.sample(x_range), rng.sample(y_range)])
        .collect::<Vec<_>>()
}


/*TERRAIN GENERATION */

fn islands(voronoi: &Voronoi, k: usize) -> Vec<usize>{
    let mut rng = rand::rng();
    let n = voronoi.cells().len();
    let mut cells: Vec<usize> = Vec::with_capacity(n); 

    for i in 0..k {

        let cell_index = rng.random_range(0..n);
        let cell = voronoi.cell(cell_index);
        //land_max_depth(voronoi, &cell, &mut cells, cell_index, &mut rng, 20);
        land(voronoi, &cell, &mut cells, cell_index, &mut rng);
    }

    cells
}

// From mainland points, recursively walk to neighbors, adding to land until some stopping condition
fn land (diagram: &Voronoi, cell: &VoronoiCell, cells: &mut Vec<usize>, cellpos: usize, rng: &mut ThreadRng) {
    //println!("Visiting cell {:?}", cellpos);
    cells.push(cellpos);

    for neighbor in cell.iter_neighbors() {    
        let tuple = diagram.cell(neighbor).site_position().clone();

        if (rng.random_range(0.0..1.0) > GAMMA) && !cells.contains(&neighbor) {
            land(diagram, &diagram.cell(neighbor), cells, neighbor, rng);
        }
    }
}


fn land_max_depth (diagram: &Voronoi, cell: &VoronoiCell, cells: &mut Vec<usize>, cellpos: usize, rng: &mut ThreadRng, depth: usize) {
    println!("Visiting cell {:?}", cellpos);
    cells.push(cellpos);

    if depth == 0 {
        println!("Max depth reached at cell {:?}", cellpos);
        return; 
    }
    for neighbor in cell.iter_neighbors() {    
        let tuple = diagram.cell(neighbor).site_position().clone();

        if (rng.random_range(0.0..1.0) > GAMMA) && !cells.contains(&neighbor) {
            land_max_depth(diagram, &diagram.cell(neighbor), cells, neighbor, rng, depth - 1);
        }
    }
}



use rand::{seq::SliceRandom};

/// Pick K site indices nearest to a random center (biased “blob”)
fn pick_island_cells(sites: &[Point], k: usize, bbox_w: f64, bbox_h: f64) -> Vec<usize> {
    let mut rng = rand::thread_rng();
    // Random center inside the bbox
    let cx = rng.gen_range(0.0..700.0);
    let cy = rng.gen_range(0.0..700.0);
    let mut idxs: Vec<usize> = (0..sites.len()).collect();

    idxs.sort_by(|&a, &b| {
        let da = (sites[a].x - cx).hypot(sites[a].y - cy);
        let db = (sites[b].x - cx).hypot(sites[b].y - cy);
        da.partial_cmp(&db).unwrap()
    });

    // Take more than k, then shuffle-drop to add wobbly edges
    let overshoot = (k as f64 * 1.15) as usize;
    let mut chosen = idxs.into_iter().take(overshoot.max(k)).collect::<Vec<_>>();
    println!("{:?}, {:?}", chosen.first(), sites[chosen[0]]);

    chosen.shuffle(&mut rng);
    chosen.truncate(k);
    chosen
}

/*
fn noisyIsland() {
    let noise = Perlin::new();
}
*/


/// Build an SVG <g> where each selected Voronoi cell is a filled polygon.
/// Use the same fill and no stroke so adjacent cells look like a single island.

/// Convenience: insert a <g> before </svg>
fn inject_group(mut svg: String, group: &str) -> String {
    if let Some(pos) = svg.rfind("</svg>") {
        svg.insert_str(pos, group);
        svg
    } else {
        // Fallback: wrap as an SVG if caller passed only fragments
        format!(r#"<svg xmlns="http://www.w3.org/2000/svg">{group}</svg>"#)
    }
}



/* RENDERING */


pub fn voronoi_to_svg(voronoi: &Voronoi, transform: &Transform, cell_indices: &[usize]) -> String {
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
        {islands}
    </svg>"#,
            width = CANVAS_SIZE,
            height = CANVAS_SIZE,
            bb_x = bounding_box_top_left.x,
            bb_y = bounding_box_top_left.y,
            bb_width = bounding_box_side,
            bb_height = bounding_box_side,
            sites = render_point(&transform, voronoi.sites(), SITE_COLOR, false, false),
            circumcenters = "",//render_point(&transform, voronoi.vertices(), CIRCUMCENTER_COLOR, true, true),
            voronoi_edges =  render_voronoi_edges(&transform, &voronoi),
            triangles = "", //render_triangles(&transform, &voronoi, false, true),
            circumcenter_circles = "",//render_circumcenters(&transform, &voronoi),
            islands = island_group_svg(voronoi, &transform, cell_indices, SAND_COLOR, 0.8)
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
            // If you have a coordinate transform, apply it here:
            // let (sx, sy) = transform((v.x, v.y));
            // pts.push(format!("{:.2},{:.2}", sx, sy));
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
