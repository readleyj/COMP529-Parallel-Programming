#include <queue>
#include <iostream>
    
    
void print_queue(std::queue<int> q)
{
  while (!q.empty())
  {
    std::cout << q.front() << " ";
    q.pop();
  }
  std::cout << std::endl;
}


void print_result(graph_t* graph, int* result, int max_depth)
{
    for (int i = 0; i < max_depth; i++) 
    {
        int *lvl = new int[graph->num_vertices];
        int idx = 0;
        for (int v = 0; v < graph->num_vertices; v++) 
        {
            if (result[v] == i) 
            {
                lvl[idx] = v;
                idx++;
            }
        }
        std::cout << "Depth " << i << "| Size " << idx << ": ";

        for (int lv = 0; lv < idx-1; lv++) 
        {
            std::cout << lvl[lv] << ", ";
        }
        std::cout << lvl[idx-1] << std::endl << std::endl;

        for (int lv = 0; lv < idx; lv++) 
        {
            int v = lvl[lv];
            int n = graph->v_adj_begin[v]; 
            std::cout << v << " | size = " << graph->v_adj_length[v] << ": ";
            while (n < graph->v_adj_begin[v] + graph->v_adj_length[v] - 1)
            {
                int neighbor = graph->v_adj_list[n];
                std::cout << neighbor << ", "; 
                n++;
            }
            std::cout << graph->v_adj_list[n] << std::endl;
        }
        std::cout << std::endl;
    }
}


int bfs_queue(graph_t *graph, int start_vertex, int *result)
{
    int num_vertices = graph->num_vertices; 
    bool *visited = new bool[num_vertices];
    fill_n(result, num_vertices, MAX_DIST);
    fill_n(visited, num_vertices, 0);

    auto start_time = Time::now();

    visited[start_vertex] = true;
    result[start_vertex] = 0;

    queue<int> next_vertices;
    next_vertices.push(start_vertex);
    int max_depth = 0;

    while (!next_vertices.empty())
    {
        int vertex = next_vertices.front();
        next_vertices.pop();

        for (
            int n = graph->v_adj_begin[vertex]; 
            n < graph->v_adj_begin[vertex] + graph->v_adj_length[vertex]; 
            n++)
        {
            int neighbor = graph->v_adj_list[n];

            if (!visited[neighbor])
            {
                visited[neighbor] = true;
                result[neighbor] = result[vertex] + 1;
                if (result[neighbor] > max_depth) max_depth = result[neighbor];
                next_vertices.push(neighbor);
            }
        }
    }

    //print_result(graph, result, max_depth+1);
    return std::chrono::duration_cast<us>(Time::now()-start_time).count();
}


int bfs_naive(graph_t *graph, int start_vertex, int *result)
{
    int num_vertices = graph->num_vertices; 
    fill_n(result, num_vertices, MAX_DIST);
    
    auto start_time = Time::now();
    
    int depth = 0;
    result[start_vertex] = depth;

    int keep_going = true;

    while (keep_going)
    {
        keep_going = false;

        for (int vertex = 0; vertex < num_vertices; vertex++)
        {
            if (result[vertex] == depth) {
                for (int n = graph->v_adj_begin[vertex]; 
                    n < graph->v_adj_begin[vertex] + graph->v_adj_length[vertex]; 
                    n++)
                {
                    int neighbor = graph->v_adj_list[n];

                    if (result[neighbor] > depth+1)
                    {
                        result[neighbor] = depth+1;
                        keep_going = true;
                    }
                }
            }
        }

        depth++;
    }

    //print_result(graph, result, depth);
    return std::chrono::duration_cast<us>(Time::now()-start_time).count();
}


int bfs_frontier(graph_t *graph, int start_vertex, int *result)
{
    int num_vertices = graph->num_vertices; 
    fill_n(result, num_vertices, MAX_DIST);

    auto start_time = Time::now();
    
    int depth = 0;
    result[start_vertex] = depth;

    int *frontier_in = new int[num_vertices];
    int *frontier_out = new int[num_vertices];
    frontier_in[0] = start_vertex;
    int front_in_size = 1;
    int front_out_size = 0;

    while (front_in_size != 0)
    {
        front_out_size = 0;

        for (int v = 0; v < front_in_size; v++)
        {
            int vertex = frontier_in[v];

            for (int n = graph->v_adj_begin[vertex]; 
                n < graph->v_adj_begin[vertex] + graph->v_adj_length[vertex]; 
                n++)
            {
                int neighbor = graph->v_adj_list[n];

                if (result[neighbor] > depth+1)
                {
                    result[neighbor] = depth+1;
                    frontier_out[front_out_size] = neighbor;
                    front_out_size++;
                }
            }
        }
        
        front_in_size = front_out_size;
        int* temp = frontier_in;
        frontier_in = frontier_out;
        frontier_out = temp;
        depth++;
    }

    //print_result(graph, result, depth);
    return std::chrono::duration_cast<us>(Time::now()-start_time).count();
}