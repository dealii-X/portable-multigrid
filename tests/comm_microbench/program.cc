// MPI latency / bandwidth microbenchmarks for the crossover study (see
// tests/poisson_cube_hybrid_gpu_cpu/PLAN.md, Phase 2B).
//
//   mpirun -n <ranks> ./program <test> [options]
//
//   test = pingpong | halo | allreduce
//   --mem host|device|staged   buffer location (default host). device = pass
//                              the device pointer to MPI (needs CUDA-aware
//                              MPI), staged = device buffer + copy to a host
//                              buffer around every MPI call
//   --group r0,r1,...          ranks that take part (pingpong: exactly 2).
//                              default: pingpong 0,1; halo and allreduce all
//   --idle spin|sleep          how the other ranks wait: blocking MPI_Barrier
//                              (spin) or MPI_Ibarrier + usleep (sleep)
//   --max-bytes N              largest message (default 16 MiB), sizes x4
//   --reps N                   timed repetitions per size (default 200)
//
// Output, rank 0, CSV: test,mem,idle,nranks,group,bytes,reps,t_min,t_median,GB_per_s
//   pingpong: t = half round trip. halo: t = one exchange of `bytes` with every
//   other group member (Isend/Irecv). allreduce: t = MPI_Allreduce of `bytes`.
// All times in seconds; only group members time, the others wait.

#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

namespace
{
  enum class Mem { host, device, staged };

  struct Options
  {
    std::string test;
    Mem         mem        = Mem::host;
    bool        sleep_idle = false;
    std::size_t max_bytes  = std::size_t(16) << 20;
    int         reps       = 200;
    std::vector<int> group;
  };

  std::vector<int>
  parse_list(const std::string &s)
  {
    std::vector<int> v;
    std::stringstream ss(s);
    std::string       item;
    while (std::getline(ss, item, ','))
      v.push_back(std::atoi(item.c_str()));
    return v;
  }

  // all ranks end a size together. spin: MPI_Barrier (busy polling inside MPI).
  // sleep: MPI_Ibarrier; non-members poll it every 100 us (blocking and
  // nonblocking collectives do not match, so members use Ibarrier too)
  void
  end_of_size(const bool sleep_idle, const bool member)
  {
    if (!sleep_idle)
      {
        MPI_Barrier(MPI_COMM_WORLD);
        return;
      }
    MPI_Request request;
    MPI_Ibarrier(MPI_COMM_WORLD, &request);
    if (member)
      {
        MPI_Wait(&request, MPI_STATUS_IGNORE);
        return;
      }
    int done = 0;
    while (!done)
      {
        MPI_Test(&request, &done, MPI_STATUS_IGNORE);
        if (!done)
          usleep(100);
      }
  }

  // send/receive buffers of one rank, in the requested memory
  struct Buffers
  {
    using DeviceView = Kokkos::View<char *>;
    using HostView   = Kokkos::View<char *, Kokkos::HostSpace>;

    Buffers(const std::size_t bytes, const int n_peers, const Mem mem) : mem(mem), bytes(bytes)
    {
      const std::size_t n = std::max<std::size_t>(bytes * std::max(n_peers, 1), 1);
      send   = DeviceView("send", n);
      recv   = DeviceView("recv", n);
      if (mem == Mem::staged)
        {
          host_send = HostView("host_send", n);
          host_recv = HostView("host_recv", n);
        }
      else if (mem == Mem::host)
        {
          host_send = HostView("host_send", n);
          host_recv = HostView("host_recv", n);
        }
      Kokkos::deep_copy(send, char(1));
      Kokkos::fence();
    }

    // pointers handed to MPI for peer slot i
    char *
    send_ptr(const int i)
    {
      return (mem == Mem::device ? send.data() : host_send.data()) + std::size_t(i) * bytes;
    }
    char *
    recv_ptr(const int i)
    {
      return (mem == Mem::device ? recv.data() : host_recv.data()) + std::size_t(i) * bytes;
    }

    // staged: device -> host before sending, host -> device after receiving
    void
    stage_out()
    {
      if (mem == Mem::staged)
        {
          Kokkos::deep_copy(host_send, send);
          Kokkos::fence();
        }
    }
    void
    stage_in()
    {
      if (mem == Mem::staged)
        {
          Kokkos::deep_copy(recv, host_recv);
          Kokkos::fence();
        }
    }

    Mem         mem;
    std::size_t bytes;
    DeviceView  send, recv;
    HostView    host_send, host_recv;
  };

  double
  median(std::vector<double> v)
  {
    std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
    return v[v.size() / 2];
  }

  const char *
  mem_name(const Mem m)
  {
    return m == Mem::host ? "host" : m == Mem::device ? "device" : "staged";
  }
} // namespace


int
main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  Kokkos::initialize(argc, argv);
  int status = 0;
  {
    int rank = 0, nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    Options opt;
    if (argc < 2)
      {
        if (rank == 0)
          std::cout << "Usage: ./program pingpong|halo|allreduce [--mem host|device|staged] "
                       "[--group r0,r1,..] [--idle spin|sleep] [--max-bytes N] [--reps N]"
                    << std::endl;
        status = 1;
      }
    else
      {
        opt.test = argv[1];
        for (int i = 2; i + 1 < argc; i += 2)
          {
            const std::string key = argv[i], val = argv[i + 1];
            if (key == "--mem")
              opt.mem = val == "device" ? Mem::device : val == "staged" ? Mem::staged : Mem::host;
            else if (key == "--group")
              opt.group = parse_list(val);
            else if (key == "--idle")
              opt.sleep_idle = val == "sleep";
            else if (key == "--max-bytes")
              opt.max_bytes = std::strtoull(val.c_str(), nullptr, 10);
            else if (key == "--reps")
              opt.reps = std::atoi(val.c_str());
          }

        if (opt.group.empty())
          {
            if (opt.test == "pingpong")
              opt.group = {0, std::min(1, nranks - 1)};
            else
              {
                opt.group.resize(nranks);
                std::iota(opt.group.begin(), opt.group.end(), 0);
              }
          }

        const auto it = std::find(opt.group.begin(), opt.group.end(), rank);
        const bool member = it != opt.group.end();
        const int  index  = member ? int(it - opt.group.begin()) : -1;
        const int  gsize  = int(opt.group.size());

        MPI_Group world_group, sub_group;
        MPI_Comm  comm = MPI_COMM_NULL;
        MPI_Comm_group(MPI_COMM_WORLD, &world_group);
        MPI_Group_incl(world_group, gsize, opt.group.data(), &sub_group);
        MPI_Comm_create_group(MPI_COMM_WORLD, sub_group, 0, &comm);

        std::string group_str;
        for (std::size_t i = 0; i < opt.group.size(); ++i)
          group_str += (i ? "+" : "") + std::to_string(opt.group[i]);

        if (rank == 0)
          std::cout << "test,mem,idle,nranks,group,bytes,reps,t_min,t_median,GB_per_s" << std::endl;

        const int reps = opt.reps, warmup = 10;
        std::vector<std::size_t> sizes;
        if (opt.test == "allreduce")
          sizes = {8, 1024, 65536};
        else
          for (std::size_t b = 8; b <= opt.max_bytes; b *= 4)
            sizes.push_back(b);

        for (const std::size_t bytes : sizes)
          {
            double t_min = 0, t_med = 0;
            if (member)
              {
                const int n_peers = opt.test == "halo" ? gsize - 1 : 1;
                Buffers   buf(bytes, n_peers, opt.mem);
                std::vector<double> times;
                std::vector<MPI_Request> req(2 * std::max(n_peers, 1));
                std::vector<double>      dbuf(std::max<std::size_t>(bytes / 8, 1), 1.0), dout(dbuf.size());

                for (int r = 0; r < reps + warmup; ++r)
                  {
                    MPI_Barrier(comm);
                    const double t0 = MPI_Wtime();
                    if (opt.test == "pingpong")
                      {
                        const int peer = index == 0 ? 1 : 0;
                        if (index == 0)
                          {
                            buf.stage_out();
                            MPI_Send(buf.send_ptr(0), int(bytes), MPI_CHAR, opt.group[peer], 0, MPI_COMM_WORLD);
                            MPI_Recv(buf.recv_ptr(0), int(bytes), MPI_CHAR, opt.group[peer], 1, MPI_COMM_WORLD,
                                     MPI_STATUS_IGNORE);
                            buf.stage_in();
                          }
                        else
                          {
                            MPI_Recv(buf.recv_ptr(0), int(bytes), MPI_CHAR, opt.group[peer], 0, MPI_COMM_WORLD,
                                     MPI_STATUS_IGNORE);
                            buf.stage_in();
                            buf.stage_out();
                            MPI_Send(buf.send_ptr(0), int(bytes), MPI_CHAR, opt.group[peer], 1, MPI_COMM_WORLD);
                          }
                      }
                    else if (opt.test == "halo")
                      {
                        buf.stage_out();
                        int n = 0, slot = 0;
                        for (int g = 0; g < gsize; ++g)
                          if (g != index)
                            {
                              MPI_Irecv(buf.recv_ptr(slot), int(bytes), MPI_CHAR, opt.group[g], 2, MPI_COMM_WORLD,
                                        &req[n++]);
                              MPI_Isend(buf.send_ptr(slot), int(bytes), MPI_CHAR, opt.group[g], 2, MPI_COMM_WORLD,
                                        &req[n++]);
                              ++slot;
                            }
                        MPI_Waitall(n, req.data(), MPI_STATUSES_IGNORE);
                        buf.stage_in();
                      }
                    else if (opt.test == "allreduce")
                      MPI_Allreduce(dbuf.data(), dout.data(), int(dbuf.size()), MPI_DOUBLE, MPI_SUM, comm);

                    double t = MPI_Wtime() - t0;
                    if (opt.test == "pingpong")
                      t *= 0.5;
                    if (r >= warmup)
                      times.push_back(t);
                  }
                // worst member decides: one number per size
                t_min = *std::min_element(times.begin(), times.end());
                t_med = median(times);
                double mx[2] = {t_min, t_med}, out[2];
                MPI_Reduce(mx, out, 2, MPI_DOUBLE, MPI_MAX, 0, comm);
                t_min = out[0];
                t_med = out[1];
              }

            // non-members wait here while the members measure; members join once done
            end_of_size(opt.sleep_idle, member);

            if (index == 0)
              {
                const double moved = opt.test == "halo" ? double(bytes) * (gsize - 1) : double(bytes);
                std::cout << opt.test << ',' << mem_name(opt.mem) << ',' << (opt.sleep_idle ? "sleep" : "spin")
                          << ',' << nranks << ',' << group_str << ',' << bytes << ',' << reps << ',' << t_min
                          << ',' << t_med << ',' << moved / t_med / 1e9 << std::endl;
              }
          }

        if (comm != MPI_COMM_NULL)
          MPI_Comm_free(&comm);
        MPI_Group_free(&sub_group);
        MPI_Group_free(&world_group);
      }
  }
  Kokkos::finalize();
  MPI_Finalize();
  return status;
}
