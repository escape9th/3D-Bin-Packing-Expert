# -*- coding: utf-8 -*-
"""
3D Bin Packing (三维装箱问题) 核心算法模块

本模块实现了基于极角点(Extreme Points)的启发式算法来解决3D装箱问题。
主要特点：
1. 物品漂浮修正(fix_point)：模拟重力，确保物品不会悬空
2. 稳定性检查(check_stable)：确保物品放置稳定
3. 支持多物品绑定、旋转、承重限制等功能
4. 计算重心分布，评估装箱平衡性
"""

from .constants import RotationType, Axis
from .auxiliary_methods import intersect, set2Decimal
import numpy as np
# 导入matplotlib用于可视化绘图
from matplotlib.patches import Rectangle, Circle
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.art3d as art3d
from collections import Counter
import copy

# 默认小数位数
DEFAULT_NUMBER_OF_DECIMALS = 0
# 物品放置的起始位置
START_POSITION = [0, 0, 0]


class Item:
    """
    物品类 - 表示待装箱的物品

    属性:
        partno: 物品唯一标识符
        name: 物品名称/类型
        typeof: 物品形状类型 ('cube'立方体 或 'cylinder'圆柱体)
        width, height, depth: 物品的宽、高、深尺寸
        weight: 物品重量
        level: 装箱优先级（数值越小优先级越高）
        loadbear: 单位面积承重能力（kg/cm²）
        updown: 是否可以上下翻转
        color: 物品颜色（用于可视化）
        rotation_type: 当前旋转类型
        position: 物品在箱子中的位置坐标
    """

    def __init__(self, partno, name, typeof, WHD, weight, level, loadbear, updown, color):
        """
        初始化物品对象

        参数:
            partno: 物品唯一标识符
            name: 物品名称
            typeof: 物品类型 ('cube' 或 'cylinder')
            WHD: 物品尺寸 (宽, 高, 深)
            weight: 物品重量
            level: 装箱优先级 (1-3, 数值越小优先级越高)
            loadbear: 单位面积承重能力（kg/cm²）
            updown: 是否可以上下翻转
            color: 物品颜色
        """
        self.partno = partno  # 物品唯一标识
        self.name = name  # 物品名称/类型
        self.typeof = typeof  # 物品形状类型
        self.width = WHD[0]  # 宽度
        self.height = WHD[1]  # 高度
        self.depth = WHD[2]  # 深度
        self.weight = weight  # 重量
        self.level = level  # 装箱优先级，值越小优先级越高
        self.loadbear = loadbear  # 单位面积承重能力（kg/cm²）
        # 圆柱体不能上下翻转
        self.updown = updown if typeof == 'cube' else False
        self.color = color  # 颜色（可视化用）
        self.rotation_type = 0  # 当前旋转类型
        self.position = START_POSITION  # 在箱子中的位置
        self.number_of_decimals = DEFAULT_NUMBER_OF_DECIMALS
        # 当前物品已经承受的上层总重量（kg），用于真实承重约束
        self.supported_weight = 0.0

    def formatNumbers(self, number_of_decimals):
        """
        将物品的尺寸和重量格式化为指定小数位数的Decimal类型

        参数:
            number_of_decimals: 保留的小数位数
        """
        self.width = set2Decimal(self.width, number_of_decimals)
        self.height = set2Decimal(self.height, number_of_decimals)
        self.depth = set2Decimal(self.depth, number_of_decimals)
        self.weight = set2Decimal(self.weight, number_of_decimals)
        self.number_of_decimals = number_of_decimals

    def string(self):
        """
        返回物品的字符串描述

        返回:
            str: 包含物品信息的字符串
        """
        return "%s(%sx%sx%s, weight: %s) pos(%s) rt(%s) vol(%s)" % (
            self.partno, self.width, self.height, self.depth, self.weight,
            self.position, self.rotation_type, self.getVolume()
        )

    def getVolume(self):
        """
        计算物品体积

        返回:
            Decimal: 物品体积（宽×高×深）
        """
        return set2Decimal(self.width * self.height * self.depth, self.number_of_decimals)

    def getMaxArea(self):
        """
        获取物品的最大面积（用于排序）

        如果允许翻转，则返回最大的两个维度的乘积

        返回:
            Decimal: 物品最大面积
        """
        a = sorted([self.width, self.height, self.depth], reverse=True) if self.updown == True else [self.width, self.height, self.depth]
        return set2Decimal(a[0] * a[1], self.number_of_decimals)

    def getDimension(self):
        """
        根据当前旋转类型获取物品的实际尺寸

        返回:
            list: [宽, 高, 深] 顺序的尺寸列表
        """
        if self.rotation_type == RotationType.RT_WHD:
            dimension = [self.width, self.height, self.depth]
        elif self.rotation_type == RotationType.RT_HWD:
            dimension = [self.height, self.width, self.depth]
        elif self.rotation_type == RotationType.RT_HDW:
            dimension = [self.height, self.depth, self.width]
        elif self.rotation_type == RotationType.RT_DHW:
            dimension = [self.depth, self.height, self.width]
        elif self.rotation_type == RotationType.RT_DWH:
            dimension = [self.depth, self.width, self.height]
        elif self.rotation_type == RotationType.RT_WDH:
            dimension = [self.width, self.depth, self.height]
        else:
            dimension = []

        return dimension


class Bin:
    """
    箱子类 - 表示装载物品的容器

    属性:
        partno: 箱子唯一标识符
        width, height, depth: 箱子的内部尺寸
        max_weight: 箱子最大承重
        corner: 箱子角落尺寸（用于预留空间）
        items: 已装入的物品列表
        fit_items: 已装入物品的空间占用记录（用于碰撞检测）
        unfitted_items: 无法装入的物品列表
        fix_point: 是否修正物品漂浮问题
        check_stable: 是否检查物品稳定性
        support_surface_ratio: 稳定性支撑面积比率
        put_type: 物品放置顺序类型
        gravity: 物品重心分布
    """

    def __init__(self, partno, WHD, max_weight, corner=0, put_type=1):
        """
        初始化箱子对象

        参数:
            partno: 箱子唯一标识符
            WHD: 箱子尺寸 (宽, 高, 深)
            max_weight: 箱子最大承重
            corner: 箱子角落预留尺寸（厘米）
            put_type: 放置顺序类型 (0=通用, 1=通用, 2=开放顶)
        """
        self.partno = partno  # 箱子编号
        self.width = WHD[0]  # 宽度
        self.height = WHD[1]  # 高度
        self.depth = WHD[2]  # 深度
        self.max_weight = max_weight  # 最大承重
        self.corner = corner  # 角落预留尺寸
        self.items = []  # 已装入的物品
        # 初始化fit_items数组，用于记录已放置物品的空间占用
        # 格式: [x_min, x_max, y_min, y_max, z_min, z_max]
        self.fit_items = np.array([[0, WHD[0], 0, WHD[1], 0, 0]])
        self.unfitted_items = []  # 无法装入的物品
        self.total_weight = 0  # 当前箱内总重量
        self.number_of_decimals = DEFAULT_NUMBER_OF_DECIMALS
        self.fix_point = False  # 是否修正物品漂浮
        self.check_stable = False  # 是否检查稳定性
        self.support_surface_ratio = 0  # 支撑面积比率
        self.put_type = put_type  # 放置顺序类型
        self.gravity = []  # 重心分布

    def formatNumbers(self, number_of_decimals):
        """
        将箱子的尺寸和最大承重格式化为指定小数位数的Decimal类型

        参数:
            number_of_decimals: 保留的小数位数
        """
        self.width = set2Decimal(self.width, number_of_decimals)
        self.height = set2Decimal(self.height, number_of_decimals)
        self.depth = set2Decimal(self.depth, number_of_decimals)
        self.max_weight = set2Decimal(self.max_weight, number_of_decimals)
        self.number_of_decimals = number_of_decimals

    def string(self):
        """
        返回箱子的字符串描述

        返回:
            str: 包含箱子信息的字符串
        """
        return "%s(%sx%sx%s, max_weight:%s) vol(%s)" % (
            self.partno, self.width, self.height, self.depth, self.max_weight,
            self.getVolume()
        )

    def getVolume(self):
        """
        计算箱子体积

        返回:
            Decimal: 箱子体积（宽×高×深）
        """
        return set2Decimal(
            self.width * self.height * self.depth, self.number_of_decimals
        )

    def getTotalWeight(self):
        """
        计算箱子中所有物品的总重量

        返回:
            Decimal: 总重量
        """
        return set2Decimal(self.total_weight, self.number_of_decimals)

    @staticmethod
    def _overlap_length(start_a, end_a, start_b, end_b):
        return max(0.0, min(float(end_a), float(end_b)) - max(float(start_a), float(start_b)))

    def _has_overlap_2d(self, rect_a, rect_b):
        return (
            self._overlap_length(rect_a[0], rect_a[1], rect_b[0], rect_b[1]) > 0 and
            self._overlap_length(rect_a[2], rect_a[3], rect_b[2], rect_b[3]) > 0
        )

    def _overlap_area_2d(self, rect_a, rect_b):
        return (
            self._overlap_length(rect_a[0], rect_a[1], rect_b[0], rect_b[1]) *
            self._overlap_length(rect_a[2], rect_a[3], rect_b[2], rect_b[3])
        )

    def _check_and_apply_loadbearing(self, item, x, y, z, w, h, d):
        """
        承重约束（真实约束）：
        1) 上层重量按接触面积比例分摊给每个下层支撑货物
        2) 每个下层货物的 loadbear 表示单位面积承重能力（kg/cm²）
        3) 根据下层货物当前朝向的顶面面积，动态换算总可承重并校验

        参数:
            item: 待放置物品
            x, y, z: 放置坐标
            w, h, d: 待放置物品尺寸

        返回:
            bool: 是否满足承重约束；满足时会更新支撑货物supported_weight
        """
        # 在底面（z=0）无需堆叠承重校验
        if z <= 0:
            return True

        supporters = []
        total_overlap_area = 0.0

        # 找到所有与待放置物品底面接触的支撑货物（同一z顶面）
        for support in self.items:
            # 装饰件不参与承重分摊
            if support.name in ('corner', 'top_clearance'):
                continue

            sx, sy, sz = map(float, support.position)
            sw, sh, sd = map(float, support.getDimension())
            support_top_z = sz + sd

            if abs(support_top_z - z) > 1e-6:
                continue

            overlap_x = min(x + w, sx + sw) - max(x, sx)
            overlap_y = min(y + h, sy + sh) - max(y, sy)
            if overlap_x > 0 and overlap_y > 0:
                overlap_area = overlap_x * overlap_y
                supporters.append((support, overlap_area, sw, sh))
                total_overlap_area += overlap_area

        # 非底面却没有有效支撑，视为不满足承重约束
        if total_overlap_area <= 0:
            return False

        item_weight = float(item.weight)

        # 先校验，不立即写回
        staged_updates = []
        for support, overlap_area, sw, sh in supporters:
            load_share = item_weight * (overlap_area / total_overlap_area)
            current_supported = float(getattr(support, 'supported_weight', 0.0))
            new_supported = current_supported + load_share

            # 下层货物单位面积承重能力（kg/cm²）
            # 在当前朝向下，按顶面面积动态换算总可承重
            top_area = sw * sh
            if top_area <= 0:
                return False

            support_density = float(support.loadbear)
            if support_density <= 0 and load_share > 0:
                return False

            support_limit = support_density * top_area
            if support_density > 0 and new_supported - support_limit > 1e-6:
                return False

            staged_updates.append((support, new_supported))

        # 全部通过后再写回
        for support, new_supported in staged_updates:
            support.supported_weight = new_supported

        return True

    def putItem(self, item, pivot, axis=None):
        """
        将物品放置到箱子中的指定位置（极角点）

        这是核心的物品放置算法，使用极角点启发式方法：
        1. 尝试物品的所有有效旋转
        2. 检查物品是否超出箱子边界
        3. 检查物品是否与已放置物品碰撞
        4. 检查重量限制
        5. 如果启用fix_point，修正物品位置防止漂浮
        6. 如果启用check_stable，检查物品稳定性

        参数:
            item: 要放置的物品
            pivot: 放置的参考点（极角点）
            axis: 放置轴向

        返回:
            bool: 如果物品成功放置返回True，否则返回False
        """
        valid_item_position = item.position
        item.position = pivot

        rotate = RotationType.ALL if item.updown == True else RotationType.Notupdown

        for rotation_index in range(0, len(rotate)):
            item.rotation_type = rotation_index
            dimension = item.getDimension()

            if (
                self.width < pivot[0] + dimension[0] or
                self.height < pivot[1] + dimension[1] or
                self.depth < pivot[2] + dimension[2]
            ):
                continue

            fit = True
            for current_item_in_bin in self.items:
                if intersect(current_item_in_bin, item):
                    fit = False
                    break

            if not fit:
                item.position = valid_item_position
                continue

            if self.total_weight + item.weight > self.max_weight:
                item.position = valid_item_position
                return False

            if self.fix_point == True:
                [w, h, d] = dimension
                [x, y, z] = [float(pivot[0]), float(pivot[1]), float(pivot[2])]

                for _ in range(3):
                    y = self.checkHeight([x, x + float(w), y, y + float(h), z, z + float(d)])
                    x = self.checkWidth([x, x + float(w), y, y + float(h), z, z + float(d)])
                    z = self.checkDepth([x, x + float(w), y, y + float(h), z, z + float(d)])

                if self.check_stable == True:
                    item_area_lower = float(dimension[0] * dimension[1])
                    support_area_upper = 0.0
                    item_rect = [x, x + float(w), y, y + float(h)]

                    for fit_item in self.fit_items:
                        if z != fit_item[5]:
                            continue
                        support_area_upper += self._overlap_area_2d(item_rect, [fit_item[0], fit_item[1], fit_item[2], fit_item[3]])

                    if item_area_lower <= 0 or support_area_upper / item_area_lower < self.support_surface_ratio:
                        four_vertices = [[x, y], [x + float(w), y], [x, y + float(h)], [x + float(w), y + float(h)]]
                        c = [False, False, False, False]
                        for fit_item in self.fit_items:
                            if z != fit_item[5]:
                                continue
                            for jdx, vertex in enumerate(four_vertices):
                                if (fit_item[0] <= vertex[0] <= fit_item[1]) and (fit_item[2] <= vertex[1] <= fit_item[3]):
                                    c[jdx] = True
                        if False in c:
                            item.position = valid_item_position
                            continue

                if not self._check_and_apply_loadbearing(item, x, y, z, float(w), float(h), float(d)):
                    item.position = valid_item_position
                    continue

                self.fit_items = np.append(self.fit_items, np.array([[x, x + float(w), y, y + float(h), z, z + float(d)]]), axis=0)
                item.position = [set2Decimal(x), set2Decimal(y), set2Decimal(z)]

            self.items.append(copy.deepcopy(item))
            self.total_weight += item.weight
            return True

        item.position = valid_item_position
        return False

    def checkDepth(self, unfix_point):
        """
        修正物品在深度(Z轴)方向的位置

        确保物品不会漂浮在空隙中，找到下方最近的支撑点

        参数:
            unfix_point: 未修正的位置坐标 [x_min, x_max, y_min, y_max, z_min, z_max]

        返回:
            float: 修正后的深度起始位置
        """
        z_ = [[0, 0], [float(self.depth), float(self.depth)]]
        top_rect = [unfix_point[0], unfix_point[1], unfix_point[2], unfix_point[3]]
        for fit_item in self.fit_items:
            bottom_rect = [fit_item[0], fit_item[1], fit_item[2], fit_item[3]]
            if self._has_overlap_2d(bottom_rect, top_rect):
                z_.append([float(fit_item[4]), float(fit_item[5])])

        top_depth = unfix_point[5] - unfix_point[4]
        z_ = sorted(z_, key=lambda z_: z_[1])
        for j in range(len(z_) - 1):
            if z_[j + 1][0] - z_[j][1] >= top_depth:
                return z_[j][1]
        return unfix_point[4]

    def checkWidth(self, unfix_point):
        """
        修正物品在宽度(X轴)方向的位置

        确保物品不会悬空在侧面，找到左侧最近的支撑边界

        参数:
            unfix_point: 未修正的位置坐标 [x_min, x_max, y_min, y_max, z_min, z_max]

        返回:
            float: 修正后的宽度起始位置
        """
        x_ = [[0, 0], [float(self.width), float(self.width)]]
        top_rect = [unfix_point[4], unfix_point[5], unfix_point[2], unfix_point[3]]
        for fit_item in self.fit_items:
            bottom_rect = [fit_item[4], fit_item[5], fit_item[2], fit_item[3]]
            if self._has_overlap_2d(bottom_rect, top_rect):
                x_.append([float(fit_item[0]), float(fit_item[1])])

        top_width = unfix_point[1] - unfix_point[0]
        x_ = sorted(x_, key=lambda x_: x_[1])
        for j in range(len(x_) - 1):
            if x_[j + 1][0] - x_[j][1] >= top_width:
                return x_[j][1]
        return unfix_point[0]

    def checkHeight(self, unfix_point):
        """
        修正物品在高度(Y轴)方向的位置

        这是解决物品漂浮问题的核心，确保物品放在其他物品或箱子底部上

        参数:
            unfix_point: 未修正的位置坐标 [x_min, x_max, y_min, y_max, z_min, z_max]

        返回:
            float: 修正后的高度起始位置
        """
        y_ = [[0, 0], [float(self.height), float(self.height)]]
        top_rect = [unfix_point[0], unfix_point[1], unfix_point[4], unfix_point[5]]
        for fit_item in self.fit_items:
            bottom_rect = [fit_item[0], fit_item[1], fit_item[4], fit_item[5]]
            if self._has_overlap_2d(bottom_rect, top_rect):
                y_.append([float(fit_item[2]), float(fit_item[3])])

        top_height = unfix_point[3] - unfix_point[2]
        y_ = sorted(y_, key=lambda y_: y_[1])
        for j in range(len(y_) - 1):
            if y_[j + 1][0] - y_[j][1] >= top_height:
                return y_[j][1]

        return unfix_point[2]

    def addCorner(self):
        """
        添加箱子角落装饰物

        用于在可视化时显示箱子角落的预留空间

        返回:
            list: 角落物品列表
        """
        if self.corner != 0:
            corner = set2Decimal(self.corner)
            corner_list = []
            for i in range(8):
                a = Item(
                    partno='corner{}'.format(i),
                    name='corner',
                    typeof='cube',
                    WHD=(corner, corner, corner),
                    weight=0,
                    level=0,
                    loadbear=0,
                    updown=True,
                    color='#000000')
                corner_list.append(a)
            return corner_list

    def putCorner(self, info, item):
        """
        放置箱子角落装饰物

        参数:
            info: 角落位置索引 (0-7)
            item: 角落物品对象
        """
        x = set2Decimal(self.width - self.corner)
        y = set2Decimal(self.height - self.corner)
        z = set2Decimal(self.depth - self.corner)
        pos = [[0, 0, 0], [0, 0, z], [0, y, z], [0, y, 0], [x, y, 0], [x, 0, 0], [x, 0, z], [x, y, z]]
        item.position = pos[info]
        self.items.append(item)

        corner = [float(item.position[0]), float(item.position[0])+float(self.corner),
                  float(item.position[1]), float(item.position[1])+float(self.corner),
                  float(item.position[2]), float(item.position[2])+float(self.corner)]

        self.fit_items = np.append(self.fit_items, np.array([corner]), axis=0)

    def clearBin(self):
        """
        清空箱子中的所有物品，重置为初始状态
        """
        self.items = []
        self.total_weight = 0
        self.fit_items = np.array([[0, self.width, 0, self.height, 0, 0]])


class Packer:
    """
    装箱器类 - 负责管理所有箱子物品的装箱操作

    核心算法流程：
    1. 对物品按体积、承重、优先级排序
    2. 对箱子按体积排序
    3. 使用极角点启发式方法放置物品
    4. 支持物品绑定、稳定性检查、重心计算等功能
    """

    def __init__(self):
        """
        初始化装箱器
        """
        self.bins = []  # 箱子列表
        self.items = []  # 待装箱物品列表
        self.unfit_items = []  # 无法装箱的物品列表
        self.total_items = 0  # 物品总数
        self.binding = []  # 物品绑定关系

    def addBin(self, bin):
        """
        添加箱子到装箱器

        参数:
            bin: 箱子对象
        """
        return self.bins.append(bin)

    def addItem(self, item):
        """
        添加物品到待装箱列表

        参数:
            item: 物品对象
        """
        self.total_items = len(self.items) + 1
        return self.items.append(item)

    def pack2Bin(self, bin, item, fix_point, check_stable, support_surface_ratio):
        """
        将单个物品装箱到指定箱子

        使用极角点启发式算法：
        1. 如果箱子为空，先放置在角落(0,0,0)
        2. 遍历三个轴向（WIDTH, HEIGHT, DEPTH）
        3. 在每个已放置物品的极角点尝试放置

        参数:
            bin: 目标箱子
            item: 要放置的物品
            fix_point: 是否修正漂浮问题
            check_stable: 是否检查稳定性
            support_surface_ratio: 支撑面积比率
        """
        fitted = False
        bin.fix_point = fix_point
        bin.check_stable = check_stable
        bin.support_surface_ratio = support_surface_ratio

        # 如果有角落设置，先添加角落
        if bin.corner != 0 and not bin.items:
            corner_lst = bin.addCorner()
            for i in range(len(corner_lst)):
                bin.putCorner(i, corner_lst[i])

        # 如果箱子为空，放置在起点(0,0,0)
        elif not bin.items:
            response = bin.putItem(item, item.position)

            if not response:
                bin.unfitted_items.append(item)
            return

        # 遍历三个轴向，在每个轴向的极角点尝试放置
        for axis in range(0, 3):
            items_in_bin = bin.items
            for ib in items_in_bin:
                pivot = [0, 0, 0]
                w, h, d = ib.getDimension()
                if axis == Axis.WIDTH:
                    # 在物品宽度方向上创建极角点
                    pivot = [ib.position[0] + w, ib.position[1], ib.position[2]]
                elif axis == Axis.HEIGHT:
                    # 在物品高度方向上创建极角点
                    pivot = [ib.position[0], ib.position[1] + h, ib.position[2]]
                elif axis == Axis.DEPTH:
                    # 在物品深度方向上创建极角点
                    pivot = [ib.position[0], ib.position[1], ib.position[2] + d]

                if bin.putItem(item, pivot, axis):
                    fitted = True
                    break
            if fitted:
                break

        if not fitted:
            bin.unfitted_items.append(item)

    def sortBinding(self, bin):
        """
        根据物品绑定关系排序

        将需要绑定在一起的物品按组排列，确保它们被连续放置

        参数:
            bin: 箱子对象（未使用）
        """
        b, front, back = [], [], []
        for i in range(len(self.binding)):
            b.append([])
            for item in self.items:
                if item.name in self.binding[i]:
                    b[i].append(item)
                elif item.name not in self.binding:
                    if len(b[0]) == 0 and item not in front:
                        front.append(item)
                    elif item not in back and item not in front:
                        back.append(item)

        min_c = min([len(i) for i in b])

        sort_bind = []
        for i in range(min_c):
            for j in range(len(b)):
                sort_bind.append(b[j][i])

        for i in b:
            for j in i:
                if j not in sort_bind:
                    self.unfit_items.append(j)

        self.items = front + sort_bind + back

    def putOrder(self):
        """
        排列箱内物品的顺序

        根据put_type类型对物品进行排序：
        - put_type=2 (开放顶容器): 按x, y, z坐标升序
        - put_type=1 (通用容器): 按y, z, x坐标升序
        """
        for i in self.bins:
            if i.put_type == 2:
                i.items.sort(key=lambda item: item.position[0], reverse=False)
                i.items.sort(key=lambda item: item.position[1], reverse=False)
                i.items.sort(key=lambda item: item.position[2], reverse=False)
            elif i.put_type == 1:
                i.items.sort(key=lambda item: item.position[1], reverse=False)
                i.items.sort(key=lambda item: item.position[2], reverse=False)
                i.items.sort(key=lambda item: item.position[0], reverse=False)

    def gravityCenter(self, bin):
        """
        计算箱子中物品的重心分布

        将箱子底面划分为四个象限，计算每个象限的重量占比
        用于评估装箱的平衡性

        参数:
            bin: 箱子对象

        返回:
            list: 四个象限的重量百分比 [Q1%, Q2%, Q3%, Q4%]
        """
        w = int(bin.width)
        h = int(bin.height)
        d = int(bin.depth)

        # 将底面划分为四个区域
        area1 = [set(range(0, w//2+1)), set(range(0, h//2+1)), 0]
        area2 = [set(range(w//2+1, w+1)), set(range(0, h//2+1)), 0]
        area3 = [set(range(0, w//2+1)), set(range(h//2+1, h+1)), 0]
        area4 = [set(range(w//2+1, w+1)), set(range(h//2+1, h+1)), 0]
        area = [area1, area2, area3, area4]

        for i in bin.items:
            x_st = int(i.position[0])
            y_st = int(i.position[1])

            # 根据旋转类型计算物品在XY平面的边界
            if i.rotation_type == 0:
                x_ed = int(i.position[0] + i.width)
                y_ed = int(i.position[1] + i.height)
            elif i.rotation_type == 1:
                x_ed = int(i.position[0] + i.height)
                y_ed = int(i.position[1] + i.width)
            elif i.rotation_type == 2:
                x_ed = int(i.position[0] + i.height)
                y_ed = int(i.position[1] + i.depth)
            elif i.rotation_type == 3:
                x_ed = int(i.position[0] + i.depth)
                y_ed = int(i.position[1] + i.height)
            elif i.rotation_type == 4:
                x_ed = int(i.position[0] + i.depth)
                y_ed = int(i.position[1] + i.width)
            elif i.rotation_type == 5:
                x_ed = int(i.position[0] + i.width)
                y_ed = int(i.position[1] + i.depth)

            x_set = set(range(x_st, int(x_ed)+1))
            y_set = set(range(y_st, int(y_ed)+1))

            # 计算每个物品属于哪个区域及其重量分配
            for j in range(len(area)):
                if x_set.issubset(area[j][0]) and y_set.issubset(area[j][1]):
                    area[j][2] += int(i.weight)
                    break
                # 部分重叠情况的处理（按重叠比例分配重量）
                elif x_set.issubset(area[j][0]) == True and y_set.issubset(area[j][1]) == False and len(y_set & area[j][1]) != 0:
                    y = len(y_set & area[j][1]) / (y_ed - y_st) * int(i.weight)
                    area[j][2] += y
                    if j >= 2:
                        area[j-2][2] += (int(i.weight) - y)
                    else:
                        area[j+2][2] += (int(i.weight) - y)
                    break
                elif x_set.issubset(area[j][0]) == False and y_set.issubset(area[j][1]) == True and len(x_set & area[j][0]) != 0:
                    x = len(x_set & area[j][0]) / (x_ed - x_st) * int(i.weight)
                    area[j][2] += x
                    if j >= 2:
                        area[j-2][2] += (int(i.weight) - x)
                    else:
                        area[j+2][2] += (int(i.weight) - x)
                    break
                elif x_set.issubset(area[j][0]) == False and y_set.issubset(area[j][1]) == False and len(y_set & area[j][1]) != 0 and len(x_set & area[j][0]) != 0:
                    all = (y_ed - y_st) * (x_ed - x_st)
                    y = len(y_set & area[0][1])
                    y_2 = y_ed - y_st - y
                    x = len(x_set & area[0][0])
                    x_2 = x_ed - x_st - x
                    area[0][2] += x * y / all * int(i.weight)
                    area[1][2] += x_2 * y / all * int(i.weight)
                    area[2][2] += x * y_2 / all * int(i.weight)
                    area[3][2] += x_2 * y_2 / all * int(i.weight)
                    break

        r = [area[0][2], area[1][2], area[2][2], area[3][2]]
        result = []
        for i in r:
            result.append(round(i / sum(r) * 100, 2))
        return result

    def pack(self, bigger_first=False, distribute_items=True, fix_point=True, check_stable=True,
             support_surface_ratio=0.75, binding=[], number_of_decimals=DEFAULT_NUMBER_OF_DECIMALS,
             sort_items=True):
        """
        装箱主函数

        算法流程：
        1. 格式化所有物品和箱子的尺寸
        2. 按体积、承重、优先级排序物品
        3. 按体积排序箱子
        4. 使用极角点启发式算法逐个放置物品
        5. 计算每个箱子的重心分布
        6. 根据distribute_items决定是否分配物品到多个箱子

        参数:
            bigger_first: 是否大物品优先（True时按体积降序）
            distribute_items: 是否分配物品到多个箱子
            fix_point: 是否修正物品漂浮问题
            check_stable: 是否检查物品稳定性
            support_surface_ratio: 支撑面积比率 (0 < ratio <= 1)
            binding: 物品绑定关系列表
            number_of_decimals: 尺寸小数位数
            sort_items: 是否按现有规则自动重排物品顺序
        """
        # 格式化尺寸和重量
        for bin in self.bins:
            bin.formatNumbers(number_of_decimals)

        for item in self.items:
            item.formatNumbers(number_of_decimals)

        self.binding = binding

        # 排序：箱子按体积排序
        self.bins.sort(key=lambda bin: bin.getVolume(), reverse=bigger_first)

        # 排序：物品按体积 -> 承重 -> 优先级排序
        if sort_items:
            self.items.sort(key=lambda item: item.getVolume(), reverse=bigger_first)
            self.items.sort(key=lambda item: item.loadbear, reverse=True)
            self.items.sort(key=lambda item: item.level, reverse=False)

        # 处理物品绑定
        if binding != []:
            self.sortBinding(bin)

        # 逐个箱子处理
        for idx, bin in enumerate(self.bins):
            # 将物品装箱到当前箱子
            for item in self.items:
                self.pack2Bin(bin, item, fix_point, check_stable, support_surface_ratio)

            # 如果有绑定关系，需要重新排序并重新装箱
            if binding != []:
                if sort_items:
                    self.items.sort(key=lambda item: item.getVolume(), reverse=bigger_first)
                    self.items.sort(key=lambda item: item.loadbear, reverse=True)
                    self.items.sort(key=lambda item: item.level, reverse=False)
                bin.items = []
                bin.unfitted_items = self.unfit_items
                bin.fit_items = np.array([[0, bin.width, 0, bin.height, 0, 0]])
                # 重新装箱
                for item in self.items:
                    self.pack2Bin(bin, item, fix_point, check_stable, support_surface_ratio)

            # 计算重心分布
#            self.bins[idx].gravity = self.gravityCenter(bin)

            # 如果启用物品分配，从待装物品列表中移除已装物品
            if distribute_items:
                for bitem in bin.items:
                    no = bitem.partno
                    for item in self.items:
                        if item.partno == no:
                            self.items.remove(item)
                            break

        # 排列物品顺序
        self.putOrder()

        # 处理无法装入的物品
        if self.items != []:
            self.unfit_items = copy.deepcopy(self.items)
            self.items = []


class Painter:
    """
    可视化画家类 - 用于绘制3D装箱结果

    使用matplotlib绘制箱子和其中物品的3D图形
    """

    def __init__(self, bins):
        """
        初始化画家对象

        参数:
            bins: 箱子对象
        """
        self.items = bins.items  # 箱内物品
        self.width = bins.width  # 箱子宽度
        self.height = bins.height  # 箱子高度
        self.depth = bins.depth  # 箱子深度

    def _plotCube(self, ax, x, y, z, dx, dy, dz, color='red', mode=2, linewidth=1, text="", fontsize=15, alpha=0.5):
        """
        绘制立方体物品

        参数:
            ax: matplotlib 3D坐标轴
            x, y, z: 立方体起点坐标
            dx, dy, dz: 立方体宽、高、深
            color: 颜色
            mode: 绘制模式 (1=线框, 2=实体)
            linewidth: 线宽
            text: 显示的文本
            fontsize: 字体大小
            alpha: 透明度
        """
        xx = [x, x, x+dx, x+dx, x]
        yy = [y, y+dy, y+dy, y, y]

        kwargs = {'alpha': 1, 'color': color, 'linewidth': linewidth}
        if mode == 1:
            # 线框模式
            ax.plot3D(xx, yy, [z]*5, **kwargs)
            ax.plot3D(xx, yy, [z+dz]*5, **kwargs)
            ax.plot3D([x, x], [y, y], [z, z+dz], **kwargs)
            ax.plot3D([x, x], [y+dy, y+dy], [z, z+dz], **kwargs)
            ax.plot3D([x+dx, x+dx], [y+dy, y+dy], [z, z+dz], **kwargs)
            ax.plot3D([x+dx, x+dx], [y, y], [z, z+dz], **kwargs)
        else:
            # 实体模式 - 绘制六个面
            p = Rectangle((x, y), dx, dy, fc=color, ec='black', alpha=alpha)
            p2 = Rectangle((x, y), dx, dy, fc=color, ec='black', alpha=alpha)
            p3 = Rectangle((y, z), dy, dz, fc=color, ec='black', alpha=alpha)
            p4 = Rectangle((y, z), dy, dz, fc=color, ec='black', alpha=alpha)
            p5 = Rectangle((x, z), dx, dz, fc=color, ec='black', alpha=alpha)
            p6 = Rectangle((x, z), dx, dz, fc=color, ec='black', alpha=alpha)
            ax.add_patch(p)
            ax.add_patch(p2)
            ax.add_patch(p3)
            ax.add_patch(p4)
            ax.add_patch(p5)
            ax.add_patch(p6)

            if text != "":
                ax.text((x+dx/2), (y+dy/2), (z+dz/2), str(text), color='black', fontsize=fontsize, ha='center', va='center')

            # 将2D矩形转换为3D
            art3d.pathpatch_2d_to_3d(p, z=z, zdir="z")
            art3d.pathpatch_2d_to_3d(p2, z=z+dz, zdir="z")
            art3d.pathpatch_2d_to_3d(p3, z=x, zdir="x")
            art3d.pathpatch_2d_to_3d(p4, z=x+dx, zdir="x")
            art3d.pathpatch_2d_to_3d(p5, z=y, zdir="y")
            art3d.pathpatch_2d_to_3d(p6, z=y+dy, zdir="y")

    def _plotCylinder(self, ax, x, y, z, dx, dy, dz, color='red', mode=2, text="", fontsize=10, alpha=0.2):
        """
        绘制圆柱体物品

        参数:
            ax: matplotlib 3D坐标轴
            x, y, z: 圆柱体起点坐标
            dx, dy: 圆柱体底面直径
            dz: 圆柱体高度
            color: 颜色
            mode: 绘制模式
            text: 显示的文本
            fontsize: 字体大小
            alpha: 透明度
        """
        # 绘制上下两个圆面
        p = Circle((x+dx/2, y+dy/2), radius=dx/2, color=color, alpha=0.5)
        p2 = Circle((x+dx/2, y+dy/2), radius=dx/2, color=color, alpha=0.5)
        ax.add_patch(p)
        ax.add_patch(p2)
        art3d.pathpatch_2d_to_3d(p, z=z, zdir="z")
        art3d.pathpatch_2d_to_3d(p2, z=z+dz, zdir="z")

        # 绘制圆柱侧面
        center_z = np.linspace(0, dz, 10)
        theta = np.linspace(0, 2*np.pi, 10)
        theta_grid, z_grid = np.meshgrid(theta, center_z)
        x_grid = dx/2 * np.cos(theta_grid) + x + dx/2
        y_grid = dy/2 * np.sin(theta_grid) + y + dy/2
        z_grid = z_grid + z
        ax.plot_surface(x_grid, y_grid, z_grid, shade=False, fc=color, alpha=alpha, color=color)

        if text != "":
            ax.text((x+dx/2), (y+dy/2), (z+dz/2), str(text), color='black', fontsize=fontsize, ha='center', va='center')

    def plotBoxAndItems(self, title="", alpha=0.2, write_num=False, fontsize=10):
        """
        绘制箱子和其中物品的3D图形

        参数:
            title: 图形标题
            alpha: 物品透明度
            write_num: 是否在物品上显示编号
            fontsize: 字体大小

        返回:
            plt: matplotlib pyplot对象
        """
        fig = plt.figure()
        axGlob = plt.axes(projection='3d')

        # 绘制箱子边框
        self._plotCube(axGlob, 0, 0, 0, float(self.width), float(self.height), float(self.depth),
                       color='black', mode=1, linewidth=2, text="")

        # 绘制每个物品
        for item in self.items:
            rt = item.rotation_type
            x, y, z = item.position
            [w, h, d] = item.getDimension()
            color = item.color
            text = item.partno if write_num else ""

            if item.typeof == 'cube':
                # 绘制立方体
                self._plotCube(axGlob, float(x), float(y), float(z), float(w), float(h), float(d),
                               color=color, mode=2, text=text, fontsize=fontsize, alpha=alpha)
            elif item.typeof == 'cylinder':
                # 绘制圆柱体
                self._plotCylinder(axGlob, float(x), float(y), float(z), float(w), float(h), float(d),
                                   color=color, mode=2, text=text, fontsize=fontsize, alpha=alpha)

        plt.title(title)
        self.setAxesEqual(axGlob)
        return plt

    def setAxesEqual(self, ax):
        """
        设置3D坐标轴等比例显示

        确保3D图形中球体显示为球体，立方体显示为立方体

        参数:
            ax: matplotlib 3D坐标轴
        """
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)

        # 计算最大范围作为图形半径
        plot_radius = 0.5 * max([x_range, y_range, z_range])

        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
