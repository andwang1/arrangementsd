#include <iostream>
#include <cmath>
#include <Eigen/Dense>
#include <vector>
#include <cassert>
#include <array>
#include <bitset>
#include <fstream>
#include <random>

// move to params
int ROOM_H = 5;
int ROOM_W = 5;
// need to bound input angle in radians

// move to params
const int discretisation = 20;
const int trajectory_length = 50;

bool VERBOSE = false;

// generates images from the trajectories fed into the function
void generate_image(std::array<Eigen::Matrix<double, discretisation, discretisation>, trajectory_length> &image_frames,
                    const std::vector<Eigen::VectorXd> trajectories)
{
    double discrete_length_x {double(ROOM_W) / discretisation};
    double discrete_length_y {double(ROOM_H) / discretisation};

    for (int i {0}; i < 2 * trajectory_length; i += 2)
    {
        // initialise image
        int image_num {i / 2};
        image_frames[image_num] = Eigen::Matrix<double, discretisation, discretisation>::Zero();
        for (auto &traj : trajectories)
        {
            double x = traj(i);
            double y = traj(i + 1);

            // flip rows and columns so x = horizontal axis
            int index_y = x / discrete_length_x;
            int index_x = y / discrete_length_y;

            image_frames[image_num](index_x, index_y) = 1;
        }
        if (VERBOSE)
        std::cout << image_frames[image_num] << std::endl;
    }
}

// assuming at most 2 impacts per frame, can add to assert in terms of dist per frame
// generates impact points and makes sure the order is correct
void generate_impact_points(std::vector<double> &wall_impacts, double previous_x, double previous_y, double previous_angle, 
                            double x_delta, double y_delta, double dist_per_frame)
{
    double epsilon = 1e-8;

    double projected_x = previous_x + x_delta;
    double projected_y = previous_y + y_delta;

    bool double_impact{false};
    // check if there will be two impacts
    if (((projected_x > ROOM_W + epsilon) && (projected_y > ROOM_H + epsilon)) || ((projected_x < 0 - epsilon) && (projected_y < 0 - epsilon)) ||
        ((projected_x > ROOM_W + epsilon) && (projected_y < 0 - epsilon)) || ((projected_x < 0 - epsilon) && (projected_y > ROOM_H + epsilon)))
    {double_impact = true;}

    double impact_1_x;
    double impact_1_y;
    double impact_2_x;
    double impact_2_y;
    double new_angle;

    if (previous_x + x_delta > ROOM_W + epsilon)
    {
        impact_1_x = ROOM_W;
        impact_1_y = previous_y + tan(previous_angle) * (ROOM_W - previous_x);
        if (double_impact)
        // one of the two conditions below must hold if there is a double impact
        {
            bool reorder{false};
            if (impact_1_y > ROOM_H + epsilon)
            // ball hits the upper y axis limit first, so recalculate
            {
                reorder = true;
                impact_1_x = previous_x + (ROOM_H - previous_y) / tan(previous_angle);
                impact_1_y = ROOM_H;
            }
            else if (impact_1_y < 0 - epsilon)
            // ball hits the lower y axis limit first
            {
                reorder = true;
                impact_1_x = previous_x - previous_y / tan(previous_angle);
                impact_1_y = 0;
            }
            else
            // no reordering required
            {
                impact_2_y = ROOM_H;
                if (previous_angle < M_PI)
                {
                    new_angle = (M_PI - previous_angle);
                    impact_2_x = ROOM_W + (ROOM_H - impact_1_y) / tan(new_angle);
                }
                else 
                {
                    new_angle = (3 * M_PI - previous_angle);
                    impact_2_x = ROOM_W - impact_1_y / tan(new_angle);
                }
            }
            if (reorder)
            // calculate second impact point after reordering
            {
                new_angle = (2 * M_PI - previous_angle);
                impact_2_x = ROOM_W;
                impact_2_y = impact_1_y + tan(new_angle) * (ROOM_W - impact_1_x);
            }
        }
    }
    else if (previous_x + x_delta < 0 + epsilon)
    {
        impact_1_x = 0;
        impact_1_y = previous_y - tan(previous_angle) * previous_x;
        if (double_impact)
        {
            bool reorder{false};
            if (impact_1_y > ROOM_H + epsilon)
            {
                reorder = true;
                impact_1_x = previous_x + (ROOM_H - previous_y) / tan(previous_angle);
                impact_1_y = ROOM_H;
            }
            else if (impact_1_y < 0 - epsilon)
            {
                reorder = true;
                impact_1_x = previous_x - previous_y / tan(previous_angle);
                impact_1_y = 0;
            }
            else
            {
                impact_2_y = 0;
                if (previous_angle < M_PI) 
                {
                    new_angle = (M_PI - previous_angle);
                    impact_2_x = (ROOM_H - impact_1_y) / tan(new_angle);
                }
                else 
                {
                    new_angle = (3 * M_PI - previous_angle);
                    impact_2_x = impact_1_y / tan(new_angle);
                }
            }
            if (reorder)
            {
                new_angle = (2 * M_PI - previous_angle);
                impact_2_x = 0;
                impact_2_y = impact_1_y + tan(new_angle) * -impact_1_x;
            }
        }
    }
    // if there is no double impact and the x limits are not exceeded, then the impact is on y    
    if ((previous_y + y_delta > ROOM_H + epsilon) && !double_impact)
    {
        impact_1_x = previous_x + (ROOM_H - previous_y) / tan(previous_angle);
        impact_1_y = ROOM_H;
    }
    if ((previous_y + y_delta < 0 - epsilon) && !double_impact)
    {
        impact_1_x = previous_x - previous_y / tan(previous_angle);
        impact_1_y = 0;
    }

    wall_impacts.push_back(impact_1_x);
    wall_impacts.push_back(impact_1_y);
    if (VERBOSE)
    {
        std::cout << "IMPACT1 X " << impact_1_x << std::endl;
        std::cout << "IMPACT1 Y " << impact_1_y << std::endl;
    }
    // this will give two impact if the points are the same, e.g. 10,10 can add a norm comparison to make it only be one
    if (double_impact)
    {
        wall_impacts.push_back(impact_2_x);
        wall_impacts.push_back(impact_2_y);
        if (VERBOSE)
        {
            std::cout << "IMPACT2 X " << impact_2_x << std::endl;
            std::cout << "IMPACT2 Y " << impact_2_y << std::endl;
        }
    }
}

// generates trajectories and the impact points
void generate_traj(Eigen::VectorXd &traj, Eigen::VectorXd &traj_impact_points, double start_x, double start_y, double angle, double dist_per_frame)
{
    // put this somewhere else? not necessarily needed
    assert (dist_per_frame < ROOM_H);
    assert (dist_per_frame < ROOM_W);

    // the wall impacts
    std::vector<double> wall_impacts;

    // assert not in wall?
    double current_x = start_x;
    double current_y = start_y;
    double current_angle = angle;

    // put starting position as first observation
    traj(0) = start_x;
    traj(1) = start_y;

    if (VERBOSE)
    {
        std::cout << "(" << start_x << ", " << start_y << ")," << std::endl;
    }

    // start at 1 since start point is the first frame
    for (int i {1}; i < trajectory_length; ++i)
    {
        if (VERBOSE)
        {std::cout << "Angle" << current_angle * 180 / M_PI << std::endl;}
        double x_delta = dist_per_frame * cos(current_angle);
        double y_delta = dist_per_frame * sin(current_angle);
        
        // save these for impact point calculations
        double previous_x = current_x;
        double previous_y = current_y;
        double previous_angle = current_angle;
        bool impact{false};
        
        current_x += x_delta;
        current_y += y_delta;

        if (current_x < 0)
        {
            impact = true;
            current_x *= -1;
            // coming from below
            if (current_angle < M_PI) {current_angle = (M_PI - current_angle);}
            else {current_angle = (3 * M_PI - current_angle);}
        }
        if (current_y < 0)
        {
            impact = true;
            current_y *= -1;
            current_angle = (2 * M_PI - current_angle);
        }
        if (current_x > ROOM_W)
        {
            impact = true;
            current_x = 2 * ROOM_W - current_x;
            if (current_angle < M_PI) {current_angle = (M_PI - current_angle);}
            else {current_angle = (3 * M_PI - current_angle);}
        }
        if (current_y > ROOM_H)
        {
            impact = true;
            current_y = 2 * ROOM_H - current_y;
            current_angle = (2 * M_PI - current_angle);
        }
        traj(i * 2) = current_x;
        traj(i * 2 + 1) = current_y;
        if (VERBOSE)
        {
            std::cout << "(" << current_x << ", " << current_y << ")," << std::endl;
        }

        if (impact)
        // pushes the impact points back onto wall_impacts
        {generate_impact_points(wall_impacts, previous_x, previous_y, previous_angle, x_delta, y_delta, dist_per_frame);}
    }
    // return the impact points
    // eigen map needs to be same type, and in constructor need the data (pointer to array data) and the sizes, for vector 1, for matrix 2
    // copy initialisation so its safe
    traj_impact_points = Eigen::Map<Eigen::VectorXd> (wall_impacts.data(), wall_impacts.size());
}

// flags all bins crossed by the trajectory and returns the number of the bin the trajectory ends in
int calculate_diversity_bins(std::bitset<discretisation * discretisation> &crossed_buckets, const Eigen::VectorXd &traj, const Eigen::VectorXd &traj_impact_points)
{
    Eigen::Vector2d start = traj.head<2>();
    Eigen::Vector2d end = traj.tail<2>();
    Eigen::Vector2d impact_pt;
    Eigen::Vector2d slope;
    Eigen::Vector2d current;

    double discrete_length_x {double(ROOM_W) / discretisation};
    double discrete_length_y {double(ROOM_H) / discretisation};

    // how many sub steps do we want to make
    double factor_divider {10};

    double factor;
    double max_factor;

    for (int i{0}; i < traj_impact_points.size() + 2; i += 2)
    {
        // not a nice way, includes the end point of the trajectory into the for loop with the above +2
        if (i >= traj_impact_points.size()) 
        {impact_pt = end;}
        else
        {impact_pt = traj_impact_points.segment(i, 2);}
        slope = impact_pt - start;
        slope.normalize();

        // create factor such that we move from bucket to bucket on one axis
        // pick the axis where there is most change to restrict
        // max factor calculation so we know when to stop
        if (abs(slope(0)) > abs(slope(1)))
        {
            factor = abs((discrete_length_x / factor_divider) / slope(0));
            max_factor = abs((impact_pt(0) - start(0)) / slope(0));
        }
        else
        {
            factor = abs((discrete_length_y / factor_divider) / slope(1));
            max_factor = abs((impact_pt(1) - start(1)) / slope(1));
        }

        if (VERBOSE)
        std::cout << "\nSTART\n" << start << "\nEND\n" << impact_pt << "\nSLOPE\n" << slope << "\nFactor " << factor << "MAXFACTOR" << max_factor << std::endl;

        for (int j{0}; j * factor < max_factor; ++j)
        // the end point of one line is included as the start point in the next line
        {
            current = start + j * factor * slope;
            
            int bucket_x = current(0) / discrete_length_x;
            int bucket_y = current(1) / discrete_length_y;
            // at the edge of the image, the bucket would be for the box outside the image, so subtract 1 to keep it inside
            if (bucket_x == discretisation) {bucket_x -= 1;}
            if (bucket_y == discretisation) {bucket_y -= 1;}
            int bucket_number = bucket_y * discretisation + bucket_x;
            
            if (VERBOSE)
            {
                std::cout << j * factor << "MAX: " << max_factor << std::endl;
                std::cout << "\nCURRENT POS\n" << current << std::endl;
                std::cout << "Bx " << bucket_x << "By " << bucket_y << "Bnum " << bucket_number << std::endl;
            }
            crossed_buckets.set(bucket_number);
        }
        start = impact_pt;
    }

    // end point of trajectory needs to be added
    int bucket_x = end(0) / discrete_length_x;
    int bucket_y = end(1) / discrete_length_y;
    if (bucket_x == discretisation) {bucket_x -= 1;}
    if (bucket_y == discretisation) {bucket_y -= 1;}
    int bucket_number = bucket_y * discretisation + bucket_x;
    crossed_buckets.set(bucket_number);

    if (VERBOSE)
    {
        std::cout << "BITS PRINTED IN REV ORDER" << std::endl;
        std::cout << crossed_buckets;
    }
    return bucket_number;
}

// start pos constrained by room width
// angle constrained by 2* M_PI
// dist 80kph @ 60fps ~ 0.35m / frame

void generate_random_data(int num_examples, bool is_random_start = true, bool is_random_dpf = true)
{
    // Will be used to obtain a seed for the random number engine
    std::random_device rd;  
    //Standard mersenne_twister_engine seeded with rd()
    std::mt19937 gen(rd()); 
    std::uniform_real_distribution<> rng_start_x(0, ROOM_W);
    std::uniform_real_distribution<> rng_start_y(0, ROOM_H);
    std::uniform_real_distribution<> rng_angle(0, 2 * M_PI);
    std::uniform_real_distribution<> rng_dpf(0, 0.35);

    std::ofstream data_file;
    data_file.open("data.txt");
    // comma seperated print
    Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ", "", "", "\n");

    double start_x, start_y, angle, dpf;

    // initialise randomly once if keeping start and dpf fixed
    if (!is_random_start)
    {
        start_x = rng_start_x(gen);
        start_y = rng_start_x(gen);
    }
    if (!is_random_dpf)
    {dpf = rng_dpf(gen);}

    Eigen::VectorXd traj(trajectory_length * 2);
    Eigen::VectorXd traj_impact_points;

    for (int i{0}; i < num_examples; ++i)
    {
        if (is_random_start)
        {
            start_x = rng_start_x(gen);
            start_y = rng_start_x(gen);
        }
        if (is_random_dpf)
        {dpf = rng_dpf(gen);}

        angle = rng_angle(gen);

        generate_traj(traj, traj_impact_points, start_x, start_y, angle, dpf);
        // first line INPUTS
        data_file << start_x << "," <<  start_y << "," << angle << "," << dpf;
        // second line is TRAJECTORY
        data_file << traj.format(CommaInitFmt);
        // third line is IMPACT
        data_file << traj_impact_points.format(CommaInitFmt) << "\n";
    }
    data_file.close();
}

int main()
{
    // generating a trajectory
    Eigen::VectorXd traj(trajectory_length * 2);
    Eigen::VectorXd traj_impact_points;
    std::vector<Eigen::VectorXd> trajectories;
    generate_traj(traj, traj_impact_points, 1, 1, 15 * M_PI / 180, 1.1);
    std::cout << "TRAJECTORY\n" << traj << std::endl;
    std::cout << "IMPACT\n" << traj_impact_points << std::endl;

    // calculating diversity score
    std::bitset<discretisation * discretisation> crossed_buckets {0};
    int end_bucket = calculate_diversity_bins(crossed_buckets, traj, traj_impact_points);
    std::cout << "END BUCKET" << end_bucket << std::endl;
    std::cout << "BUCKETS\n" << crossed_buckets << std::endl;

    // creating image frames
    trajectories.push_back(traj);
    generate_traj(traj, traj_impact_points, 3, 3, 135 * M_PI / 180, 2.6);
    trajectories.push_back(traj);
    std::array<Eigen::Matrix<double, discretisation, discretisation>, trajectory_length> image_frames;
    generate_image(image_frames, trajectories);

    // Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ", "", "", "\n");
    // std::cout << "TEST" << traj.format(CommaInitFmt);
    // writing random trajectories to file
    generate_random_data(200000, true, true);

    return 0;
}

