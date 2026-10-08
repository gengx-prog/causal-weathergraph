# 补充数据需求与官方链接

核验日期：2026年9月30日。用途：为 Major Revision 选择真正缺少的数据，并将下载入口对应到审稿问题。

**建议先完成现有数据能够支持的统计修正，再按明确目标补充少量物理控制变量或一套卫星云资料。** 南北半球直接差异检验本身不缺数据。以下优先级和变量组合是结合审稿意见提出的方案，并非审稿人要求下载全部产品，也不代表相应实验已完成。

## 已有数据和需要新增数据的边界

本地已具备1979–2025年、00/06/12/18 UTC、64×32网格的850 hPa温度、比湿、u/v风分量，以及整层总云量，共68,668个时刻。已有源文件及训练专属预处理可以继续使用。87个NetCDF文件头的核查没有在该数据集合发现其他垂直层、地表气压、位势高度、垂直速度或独立卫星云资料；名为`pressure_850`的文件实际只含850 hPa的t/q/u/v。

| 审稿问题 | 是否需要新数据 | 建议处理 |
| --- | --- | --- |
| 南北半球直接季节差异、纬度交互：R2-3、R1-D5 | 不需要 | 已有纬度、月份与序列；补正式交互或差值检验。各半球冬夏差已估计，跨半球差异尚未检验。 |
| HAC/FDR校准、完整bootstrap、随机图、方法消融、CaStLe响应窗对齐：R2-1/2/5、R3-1/2/3/7 | 不需要新增观测 | 需要改进统计设计和重跑；新气象字段不能代替这些工作。 |
| 外部环流状态：R2-3、R3-4 | 选择新增指数或场的方案时需要 | 可先选适用的AO/AAO等指数；若用z500或海平面气压定义六小时空间状态，则需补这些场。已有提前风场也可探索分类，需论证适用性。 |
| 遗漏共同驱动与垂直解释：R3-6、R1-I2/I3 | 建议少量补充 | 优先垂直速度、环流场和地表气压，按物理假设加入控制。 |
| 独立云资料验证：R1-I3/T3；R2-5的扩展 | 需要外部资料 | 选择一套符合目标时间尺度的卫星产品，不必同时下载全部候选。 |
| 三维输送、柱水汽收支：R2-4、R3-5 | 保留强机制主张时需要 | 现有u/v/qu/qv仅为850 hPa代理；可补柱积分通量，完整收支则还需储量、源汇及时间区间信息。 |

上表的审稿编号沿用已经生成的逐条答复文档。外部资料验证和完整机制分析中的部分建议属于加强证据的扩展，不能改写为审稿人的逐字强制要求。

## 优先补充的ERA5字段

两个官方入口足以覆盖主要补缺：

- [ERA5小时压力层数据](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels?tab=overview)：1940年至今，全球小时数据，37个压力层，再分析发布网格0.25°。
- [ERA5小时单层数据](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview)：地表、整层积分及云量相关字段；single level不表示全部变量都在地面。

**建议首批为omega700/500、z500、t700、地表气压和海平面气压；若只做一个小型控制实验，可先取一个omega层加一个环流场，并配地表气压用于地形处理。** 海洋子分析再加SST。研究时段仍取1979–2025；瞬时场统一为00/06/12/18 UTC。层次和范围是拟议设计，应在查看新结果前冻结。

| 要补什么 | CDS变量标识 | 如何支持审稿答复 |
| --- | --- | --- |
| 700/500 hPa垂直速度 | `vertical_velocity` | 增加抬升/下沉相关控制。单位Pa/s，是压力坐标omega，不能当作水平风速W或m/s垂直速度。 |
| 500 hPa位势 | `geopotential` | 环流背景或训练期拟合的状态分类；单位m²/s²，需要高度时转换为z/g。 |
| 700 hPa温度 | `temperature` | 低层垂直热力结构敏感性；本地850层已有。仅补T700不等于完整LTS/EIS计算。 |
| 地表气压 | `surface_pressure` | 地形和地下压力层筛选；压力层p大于当地地表气压时须按预定义方案处理。 |
| 海平面气压 | `mean_sea_level_pressure` | 大尺度环流控制或分类；与地表气压用途不同。 |
| 海表温度，可选 | `sea_surface_temperature` | 海洋云关系的边界条件；不能为陆地区域强填SST。 |

变量定义见[ECMWF当前ERA5文档](https://confluence.ecmwf.int/spaces/CKB/pages/76414402/ERA5%2Bdata%2Bdocumentation)。选控制变量还需要先明确因果时序，避免把中介或响应的后果自动当成去混杂变量；增加字段只能检验敏感性，不能证明不存在隐藏共同驱动。

需要柱输送时，在单层入口选择`vertical_integral_of_eastward_water_vapour_flux`和`vertical_integral_of_northward_water_vapour_flux`；需要分云层响应时选择`low_cloud_cover`、`medium_cloud_cover`、`high_cloud_cover`。这些仍是ERA5变量，不能充当独立观测验证。完整水汽预算还需柱水汽变化、蒸发、降水及通量辐散，不能只凭两列向量宣称收支闭合。[变量与时间语义说明](https://confluence.ecmwf.int/spaces/CKB/pages/76414402/ERA5%2Bdata%2Bdocumentation)

若目标确实需要原生137混合模型层，可用[Complete ERA5入口](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-complete?tab=overview)；这属于后续机制扩展，当前不建议先下载全球全层数据。模型层需配合[官方压力与位势计算方法](https://confluence.ecmwf.int/spaces/CKB/pages/158636068/ERA5%2Bcompute%2Bpressure%2Band%2Bgeopotential%2Bon%2Bmodel%2Blevels%2Bgeopotential%2Bheight%2Band%2Bgeometric%2Bheight)。

## 可直接获取的外部环流指数

| 指数及官方说明 | 数据直链 | 适用范围和限制 |
| --- | --- | --- |
| [NOAA CPC AO](https://www.cpc.ncep.noaa.gov/products/precip/CWlink/daily_ao_index/ao.shtml) | [下载逐日AO CSV](https://ftp.cpc.ncep.noaa.gov/cwlinks/norm.daily.ao.cdas.z1000.19500101_current.csv) | 1950年起；北半球环状模态，基于1000 hPa高度场。 |
| [NOAA CPC AAO](https://www.cpc.ncep.noaa.gov/products/precip/CWlink/daily_ao_index/aao/aao.shtml) | [下载逐日AAO CSV](https://ftp.cpc.ncep.noaa.gov/cwlinks/norm.daily.aao.cdas.z700.19790101_current.csv) | 1979年起；南半球环状模态，基于700 hPa高度场。 |
| [NOAA CPC NAO](https://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/nao.shtml) | [下载逐日NAO CSV](https://ftp.cpc.ncep.noaa.gov/cwlinks/norm.daily.nao.cdas.z500.19500101_current.csv) | 1950年起；适合北大西洋区域问题，不能代表全部全球环流。 |
| [NOAA PSL Niño 3.4](https://psl.noaa.gov/data/timeseries/month/Nino34/) | [下载逐月Niño 3.4 CSV](https://psl.noaa.gov/data/timeseries/month/data/nino34.long.anom.csv) | HadISST1.1版本，1870年起；月尺度热带太平洋背景，不是六小时状态观测。 |

上述四条直链已返回HTTP 200，并核实文件开头对应字段。正式分析仍需检查全时段缺测、冻结文件哈希与版本。若文件名更新，可从[CPC历史指数目录](https://ftp.cpc.ncep.noaa.gov/cwlinks/)重新定位。

**建议做法：**按物理区域先选AO/AAO等适用指标，以响应发生日前一日信息定义状态，训练期冻结分位阈值，再映射评价期。同一天标签重复不能增加独立样本量，推断需保留日内与跨日依赖。若采用Niño 3.4，建议使用此前已结束月份的背景量，不插值制造六小时变化。严格实时预测还需核查历史发布时间。

AO和AAO的投影层及定义不同，不能直接比较两个指数的绝对大小。提供者预设的空间模式和基期不等于本项目训练期拟合；“外部”只表示不按本项目当期响应直接分组，并不保证因果外生性或观测来源完全独立。

## 卫星云资料和气溶胶备选

若要验证2019–2025共同年代的短时标关系，优先试取CERES小时云产品；若要历史六小时可比采样，可选ISCCP HGG。日/月产品应另设相同尺度的分析。以下五类是备选，不需要全部下载。

| 产品与官方说明 | 下载入口 | 本研究如何使用及主要限制 |
| --- | --- | --- |
| [CERES SYN1deg](https://ceres.larc.nasa.gov/data/#syn1deg-level-3)，小时云与辐射 | [Earthdata集合](https://search.earthdata.nasa.gov/search/granules?p=C3181056140-LARC_CLOUD)；[官方裁剪工具](https://ceres-tool.larc.nasa.gov/ord-tool/jsp/SYN1degEd42Selection.jsp) | 2000年3月起，1°、小时；可用于原评价年代的跨云产品复核。含观测间插值及部分模式计算，不能把每小时都当作独立直接观测。 |
| [NOAA ISCCP H-series](https://www.ncei.noaa.gov/products/international-satellite-cloud-climatology)，选HGG Basic | [公开文件目录](https://www.ncei.noaa.gov/data/international-satellite-cloud-climate-project-isccp-h-series-data/access/isccp-basic/hgg/)；[THREDDS](https://www.ncei.noaa.gov/thredds/catalog/satellite/cdr-isccp-h.html) | 1°、三小时瞬时；正式CDR为1983年7月–2017年6月。不覆盖2019–2025评价期，需要另定历史共同期。HGH是月内逐时刻平均，不能替代HGG。 |
| [MERRA-2 M2T1NXAER](https://disc.gsfc.nasa.gov/datasets/M2T1NXAER_5.12.4/summary)，气溶胶控制 | [Earthdata集合](https://search.earthdata.nasa.gov/search/granules?p=C1276812830-GES_DISC)；[官方虚拟目录](https://cmr.earthdata.nasa.gov/virtual-directory/collections/C1276812830-GES_DISC) | 1980年起，0.5°纬度×0.625°经度，小时平均，V5.12.4。可选AOT550做遗漏驱动敏感性；是再分析，不是独立云观测，也不覆盖1979年。 |
| [CDS云属性](https://cds.climate.copernicus.eu/datasets/satellite-cloud-properties?tab=overview)，优先CLARA-A3 | [下载筛选页](https://cds.climate.copernicus.eu/datasets/satellite-cloud-properties?tab=download) | CLARA-A3从1979年起，0.25°日/月。当前集合元数据终点为2025年6月，不能声称已核实完整2025年。适合长期日尺度敏感性。 |
| [MODIS日产品](https://atmosphere-imager.gsfc.nasa.gov/products/daily/documentation)，次选 | [Terra MOD08_D3 6.1](https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/61/MOD08_D3/)；[Aqua MYD08_D3 6.1](https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/61/MYD08_D3/) | Terra从2000年、Aqua从2002年起，1°日统计。适合日尺度云物理复核；极轨采样不能恢复完整六小时云场，与CERES部分输入重叠。 |

**版本与配准中必须保留的区别：**

- CERES本次核实的[CMR集合](https://cmr.earthdata.nasa.gov/search/concepts/C3181056140-LARC_CLOUD.umm_json)为Terra–Aqua–NOAA20 Edition4B，但订购工具标题为Ed4.2；须以实际集合和文件锁定版本，不擅自认定两个标签相同。检查2022年平台转换，并让双方的六小时取样/平均含义一致。
- ISCCP另有2017年7月–2018年12月的临时ICDR，处理与正式CDR有差别；它不在此次核实的HGG正式目录末端，不应无标记拼接。[NCEI产品说明](https://www.ncei.noaa.gov/products/international-satellite-cloud-climatology)
- MERRA-2小时均值时间戳在半点，例如00:30，不能不经定义就与ERA5整点瞬时量相配。加入气溶胶的模型和不加气溶胶的模型，应都限定同一1980年以后的有效样本。旧服务器正在迁移，优先从集合入口获取当前地址。[官方集合定义](https://cmr.earthdata.nasa.gov/search/concepts/C1276812830-GES_DISC.umm_json)
- CDS云目录同时包含CLARA和CCI。其CCI分支为1995–2012以及2017年以后，存在间断与传感器偏移问题；不能把目录标题“1979至今”解释为CCI连续覆盖。日/月云量不能靠复制或插值变成六小时信息。[CDS官方目录元数据](https://cds.climate.copernicus.eu/api/catalogue/v1/collections/satellite-cloud-properties)

这些产品能检验结论对测量来源的敏感性，但不是因果真值；不同产品可能共享卫星或辅助资料。外部验证应包含普通和未选择候选、反向及负对照，不能只检查原先最强或最有利的关系。

## 下载顺序和统一处理方案

1. **先处理不缺数据的实验。** 南北半球正式比较、纬度交互、CaStLe响应窗对齐、统计校准及bootstrap无需等待新数据。
2. **选择一项主要数据扩展。** 针对状态选择问题，可先取适用AO/AAO指数；针对遗漏驱动，可先取小型ERA5控制集；针对云测量来源，可先选CERES。需要解决的问题不同，不应将它们混为一份必须全部完成的最小包。
3. **先做读取与单位试点。** 建议在训练期2010年1月及预设有限区域验证接口、维度、缺测和映射；试点用于数据工程，不据此调整方向假设。通过后，再按既定区域/时段分月下载所需变量。卫星验证使用预先定义的共同年代，并明确其与原1979–2025主分析的差别。
4. **固定时间与空间含义。** 瞬时场可选四个UTC时刻；六小时预算需要保留小时通量并正确聚合，抽取四个小时平均值不等于四个六小时均值。训练气候态、缩放、状态阈值仍只用1979–2018；2019–2025不用于重新挑选阈值。缺测形成的空洞不可压缩后当作连续时序。
5. **核对既有拼接和重网格。** 早期数据来自WeatherBench2，2023年1月11日起接CDS；新增场须协调网格中心和重映射。先取少量重叠样本核对即可，不建议重下全部已有五类场。[WeatherBench2官方指南](https://weatherbench2.readthedocs.io/en/latest/data-guide.html)说明64×32产品采用保守重网格；CDS请求同样的网格间距不自动复现该算子。
6. **如需真正新的时间评价，单独冻结方案。** 对尚未使用过的后续数据预设模型、目标、时间窗后再评价；重下已分析过的2019–2025不会成为新的未见测试集。若选2026年可用部分期，要明确未满全年及ERA5T可能后续修订的状态，不能将其混入旧全期主结果。[ERA5更新与最终版说明](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels?tab=overview)

精确下载量取决于区域、网格、变量和层数，需先通过试点测量估算。当前最小建议不包含全球全层原生分辨率下载。

## 账户要求和本次交付状态

NOAA指数CSV和ISCCP Basic目录可以公开访问。ERA5及CDS云产品需CDS账户、接受所选数据条款，API下载按[官方配置页](https://cds.climate.copernicus.eu/how-to-api)使用个人凭据；下载表单可以生成准确的请求代码。NASA CERES、MERRA-2和MODIS科学文件通常需Earthdata认证，部分服务还需应用授权，以具体下载入口提示为准。

这里提供的是稳定官方入口、集合链接和可验证的指数直链。CDS按需检索的文件地址在提交请求后生成，不能预先写成已经可下载的特定文件链接。本次完成了本地缺口审计、官方元数据检索和链接整理，没有提交大批量下载任务或运行新实验，也没有把登录页可访问当作科学文件下载成功。

同目录的`官方数据链接清单.csv`可用Excel打开；`sources.json`保留来源范围、访问要求和核验日期。HTML、Word和PDF版本用于浏览说明。实际获取时应另存原文件、请求参数、版本、哈希、许可与引用要求，并在最终参考文献限额内妥善安排必要的数据引用。
