#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/stroganov_m_HorizGaus3x3/include/ops_seq.hpp"

TEST(stroganov_m_HorizGaus3x3_seq, test_10_1) {
  constexpr size_t W = 10;
  constexpr size_t H = 10;
  std::vector<double> src(W * H, 1.0);
  std::vector<double> dst(W * H, 1.0);
  std::vector<double> expect(W * H, 1.0);
  std::vector<int> mask = {1, 2, 1};

  for (size_t y = 0; y < H; ++y) {
    for (size_t x = 0; x < W; ++x) {
      if (x == 0 || x == W - 1) {
        expect[y * W + x] = 0.75;
      } else {
        expect[y * W + x] = 1.0;
      }
    }
  }

  auto data = std::make_shared<ppc::core::TaskData>();
  data->inputs.emplace_back(reinterpret_cast<uint8_t*>(src.data()));
  data->inputs.emplace_back(reinterpret_cast<uint8_t*>(mask.data()));
  data->inputs_count.emplace_back(src.size());
  data->outputs.emplace_back(reinterpret_cast<uint8_t*>(dst.data()));
  data->outputs_count.emplace_back(dst.size());

  stroganov_m_horiz_gaus3x3_seq::ImageFilterSequential filter(data);

  ASSERT_TRUE(filter.Validation());

  filter.PreProcessing();
  filter.Run();
  filter.PostProcessing();

  for (size_t y = 0; y < H; ++y) {
    for (size_t x = 0; x < W; ++x) {
      ASSERT_NEAR(dst[y * W + x], expect[y * W + x], 1e-5);
    }
  }
}

TEST(stroganov_m_HorizGaus3x3_seq, test_vertical_lines) {
  constexpr size_t W = 10;
  constexpr size_t H = 10;
  std::vector<double> src(W * H, 0.0);
  std::vector<double> dst(W * H, 0.0);
  std::vector<double> expect(W * H, 0.0);
  std::vector<int> mask = {1, 2, 1};

  for (size_t y = 0; y < H; ++y) {
    src[y * W + 2] = 1.0;
    src[y * W + 7] = 1.0;
  }

  for (size_t y = 0; y < H; ++y) {
    expect[y * W + 1] = 0.25;
    expect[y * W + 2] = 0.5;
    expect[y * W + 3] = 0.25;
    expect[y * W + 6] = 0.25;
    expect[y * W + 7] = 0.5;
    expect[y * W + 8] = 0.25;
  }

  auto data = std::make_shared<ppc::core::TaskData>();
  data->inputs.emplace_back(reinterpret_cast<uint8_t*>(src.data()));
  data->inputs.emplace_back(reinterpret_cast<uint8_t*>(mask.data()));
  data->inputs_count.emplace_back(src.size());
  data->outputs.emplace_back(reinterpret_cast<uint8_t*>(dst.data()));
  data->outputs_count.emplace_back(dst.size());

  stroganov_m_horiz_gaus3x3_seq::ImageFilterSequential filter(data);

  ASSERT_TRUE(filter.Validation());

  filter.PreProcessing();
  filter.Run();
  filter.PostProcessing();

  for (size_t y = 0; y < H; ++y) {
    for (size_t x = 0; x < W; ++x) {
      ASSERT_NEAR(dst[y * W + x], expect[y * W + x], 1e-5);
    }
  }
}

TEST(stroganov_m_HorizGaus3x3_seq, test_horizontal_lines) {
  constexpr size_t W = 10;
  constexpr size_t H = 10;
  std::vector<double> src(W * H, 0.0);
  std::vector<double> dst(W * H, 0.0);
  std::vector<double> expect(W * H, 0.0);
  std::vector<int> mask = {1, 2, 1};

  for (size_t x = 0; x < W; ++x) {
    src[2 * W + x] = 1.0;
    src[7 * W + x] = 1.0;
  }

  expect[2 * W] = expect[3 * W - 1] = expect[7 * W] = expect[8 * W - 1] = 0.75;
  for (size_t x = 1; x + 1 < W; ++x) {
    expect[2 * W + x] = 1.0;
    expect[7 * W + x] = 1.0;
  }

  auto data = std::make_shared<ppc::core::TaskData>();
  data->inputs.emplace_back(reinterpret_cast<uint8_t*>(src.data()));
  data->inputs.emplace_back(reinterpret_cast<uint8_t*>(mask.data()));
  data->inputs_count.emplace_back(src.size());
  data->outputs.emplace_back(reinterpret_cast<uint8_t*>(dst.data()));
  data->outputs_count.emplace_back(dst.size());

  stroganov_m_horiz_gaus3x3_seq::ImageFilterSequential filter(data);

  ASSERT_TRUE(filter.Validation());

  filter.PreProcessing();
  filter.Run();
  filter.PostProcessing();

  for (size_t y = 0; y < H; ++y) {
    for (size_t x = 0; x < W; ++x) {
      ASSERT_NEAR(dst[y * W + x], expect[y * W + x], 1e-5);
    }
  }
}

TEST(stroganov_m_HorizGaus3x3_seq, test_empty_image) {
  constexpr size_t W = 10;
  constexpr size_t H = 10;
  std::vector<double> src(W * H, 0.0);
  std::vector<double> dst(W * H, 0.0);
  std::vector<double> expect(W * H, 0.0);
  std::vector<int> mask = {1, 2, 1};

  auto data = std::make_shared<ppc::core::TaskData>();
  data->inputs.emplace_back(reinterpret_cast<uint8_t*>(src.data()));
  data->inputs.emplace_back(reinterpret_cast<uint8_t*>(mask.data()));
  data->inputs_count.emplace_back(src.size());
  data->outputs.emplace_back(reinterpret_cast<uint8_t*>(dst.data()));
  data->outputs_count.emplace_back(dst.size());

  stroganov_m_horiz_gaus3x3_seq::ImageFilterSequential filter(data);

  ASSERT_TRUE(filter.Validation());

  filter.PreProcessing();
  filter.Run();
  filter.PostProcessing();

  for (size_t y = 0; y < H; ++y) {
    for (size_t x = 0; x < W; ++x) {
      ASSERT_NEAR(dst[y * W + x], expect[y * W + x], 1e-5);
    }
  }
}

TEST(stroganov_m_HorizGaus3x3_seq, test_sharp_edges) {
  constexpr size_t W = 10;
  constexpr size_t H = 10;
  std::vector<double> src(W * H, 0.0);
  std::vector<double> dst(W * H, 0.0);
  std::vector<double> expect(W * H, 0.0);
  std::vector<int> mask = {1, 2, 1};

  for (size_t y = 0; y < H; ++y) {
    for (size_t x = 0; x < W; ++x) {
      src[y * W + x] = (x < W / 2) ? 0.0 : 1.0;
    }
  }

  for (size_t y = 0; y < H; ++y) {
    expect[y * W + 4] = 0.25;
    expect[y * W + 5] = 0.75;
    expect[y * W + 6] = 1.0;
    expect[y * W + 7] = 1.0;
    expect[y * W + 8] = 1.0;
    expect[y * W + 9] = 0.75;
  }

  auto data = std::make_shared<ppc::core::TaskData>();
  data->inputs.emplace_back(reinterpret_cast<uint8_t*>(src.data()));
  data->inputs.emplace_back(reinterpret_cast<uint8_t*>(mask.data()));
  data->inputs_count.emplace_back(src.size());
  data->outputs.emplace_back(reinterpret_cast<uint8_t*>(dst.data()));
  data->outputs_count.emplace_back(dst.size());

  stroganov_m_horiz_gaus3x3_seq::ImageFilterSequential filter(data);

  ASSERT_TRUE(filter.Validation());

  filter.PreProcessing();
  filter.Run();
  filter.PostProcessing();

  for (size_t y = 0; y < H; ++y) {
    for (size_t x = 0; x < W; ++x) {
      ASSERT_NEAR(dst[y * W + x], expect[y * W + x], 1e-5);
    }
  }
}

TEST(stroganov_m_HorizGaus3x3_seq, test_smooth_gradient) {
  constexpr size_t W = 10;
  constexpr size_t H = 10;
  std::vector<double> src(W * H, 0.0);
  std::vector<double> dst(W * H, 0.0);
  std::vector<double> expect(W * H, 0.0);
  std::vector<int> mask = {1, 2, 1};

  for (size_t y = 0; y < H; ++y) {
    for (size_t x = 0; x < W; ++x) {
      double val = static_cast<double>(x) / (W - 1);
      src[y * W + x] = val;
      expect[y * W + x] = val;
    }
  }

  for (size_t y = 0; y < H; ++y) {
    expect[y * W] = 0.03;
    expect[y * W + W - 1] = 0.72;
  }

  auto data = std::make_shared<ppc::core::TaskData>();
  data->inputs.emplace_back(reinterpret_cast<uint8_t*>(src.data()));
  data->inputs.emplace_back(reinterpret_cast<uint8_t*>(mask.data()));
  data->inputs_count.emplace_back(src.size());
  data->outputs.emplace_back(reinterpret_cast<uint8_t*>(dst.data()));
  data->outputs_count.emplace_back(dst.size());

  stroganov_m_horiz_gaus3x3_seq::ImageFilterSequential filter(data);

  ASSERT_TRUE(filter.Validation());

  filter.PreProcessing();
  filter.Run();
  filter.PostProcessing();

  for (size_t y = 0; y < H; ++y) {
    for (size_t x = 0; x < W; ++x) {
      ASSERT_NEAR(dst[y * W + x], expect[y * W + x], 0.5);
    }
  }
}

TEST(stroganov_m_HorizGaus3x3_seq, test_all_max) {
  constexpr size_t W = 10;
  constexpr size_t H = 10;
  std::vector<double> src(W * H, 255.0);
  std::vector<double> dst(W * H, 0.0);
  std::vector<double> expect(W * H, 255.0);
  std::vector<int> mask = {1, 2, 1};

  for (size_t y = 0; y < H; ++y) {
    expect[y * W] = 191.25;
    expect[y * W + W - 1] = 191.25;
  }

  auto data = std::make_shared<ppc::core::TaskData>();
  data->inputs.emplace_back(reinterpret_cast<uint8_t*>(src.data()));
  data->inputs.emplace_back(reinterpret_cast<uint8_t*>(mask.data()));
  data->inputs_count.emplace_back(src.size());
  data->outputs.emplace_back(reinterpret_cast<uint8_t*>(dst.data()));
  data->outputs_count.emplace_back(dst.size());

  stroganov_m_horiz_gaus3x3_seq::ImageFilterSequential filter(data);

  ASSERT_TRUE(filter.Validation());

  filter.PreProcessing();
  filter.Run();
  filter.PostProcessing();

  for (size_t y = 0; y < H; ++y) {
    for (size_t x = 0; x < W; ++x) {
      ASSERT_NEAR(dst[y * W + x], expect[y * W + x], 1e-5);
    }
  }
}

TEST(stroganov_m_HorizGaus3x3_seq, test_random_mean_invariance) {
  constexpr size_t W = 100;
  constexpr size_t H = 100;

  std::vector<double> src(W * H);
  std::random_device rd;
  std::mt19937 rng(rd());
  std::uniform_real_distribution<double> dist(0.0, 255.0);

  for (double& v : src) {
    v = dist(rng);
  }

  std::vector<double> dst(W * H, 0.0);
  std::vector<int> mask = {1, 2, 1};

  auto data = std::make_shared<ppc::core::TaskData>();
  data->inputs.emplace_back(reinterpret_cast<uint8_t*>(src.data()));
  data->inputs.emplace_back(reinterpret_cast<uint8_t*>(mask.data()));
  data->inputs_count.emplace_back(src.size());
  data->outputs.emplace_back(reinterpret_cast<uint8_t*>(dst.data()));
  data->outputs_count.emplace_back(dst.size());

  stroganov_m_horiz_gaus3x3_seq::ImageFilterSequential filter(data);

  ASSERT_TRUE(filter.Validation());

  filter.PreProcessing();
  filter.Run();
  filter.PostProcessing();

  double avg_in = std::accumulate(src.begin(), src.end(), 0.0) / static_cast<double>(src.size());
  double avg_out = std::accumulate(dst.begin(), dst.end(), 0.0) / static_cast<double>(dst.size());

  ASSERT_NEAR(avg_in, avg_out, 1.0);
}
