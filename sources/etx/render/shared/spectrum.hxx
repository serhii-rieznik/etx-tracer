#pragma once

#include <etx/render/shared/base.hxx>

namespace etx {

namespace spectrum {

constexpr float kUndefinedWavelength = -1.0f;

constexpr uint32_t RGBResponseShortestWavelength = 390u;
constexpr uint32_t RGBResponseLongestWavelength = 780u;
constexpr uint32_t RGBResponseWavelengthCount = RGBResponseLongestWavelength - RGBResponseShortestWavelength + 1u;

constexpr uint32_t ShortestWavelength = 390u;
constexpr uint32_t WavelengthCount = 441u;
constexpr uint32_t LongestWavelength = ShortestWavelength + WavelengthCount - 1u;

constexpr float kShortestWavelength = static_cast<float>(ShortestWavelength);
constexpr float kLongestWavelength = static_cast<float>(LongestWavelength);
constexpr float kWavelengthCount = static_cast<float>(WavelengthCount);

constexpr float kRGBResponseShortestWavelength = static_cast<float>(RGBResponseShortestWavelength);
constexpr float kRGBResponseLongestWavelength = static_cast<float>(RGBResponseLongestWavelength);
constexpr float kRGBResponseWavelengthCount = static_cast<float>(RGBResponseWavelengthCount);

constexpr const float3 spectral_xyz(uint32_t i) {
  ETX_ASSERT(i < WavelengthCount);

  constexpr const float3 kCIE2006[WavelengthCount] = {{0.00295242f, 0.0004076779f, 0.01318752f}, {0.003577275f, 0.0004977769f, 0.01597879f},
    {0.004332146f, 0.0006064754f, 0.01935758f}, {0.005241609f, 0.000737004f, 0.02343758f}, {0.006333902f, 0.0008929388f, 0.02835021f}, {0.007641137f, 0.001078166f, 0.03424588f},
    {0.009199401f, 0.001296816f, 0.04129467f}, {0.01104869f, 0.001553159f, 0.04968641f}, {0.01323262f, 0.001851463f, 0.05962964f}, {0.01579791f, 0.002195795f, 0.07134926f},
    {0.01879338f, 0.002589775f, 0.08508254f}, {0.02226949f, 0.003036799f, 0.1010753f}, {0.02627978f, 0.003541926f, 0.1195838f}, {0.03087862f, 0.004111422f, 0.1408647f},
    {0.0361189f, 0.004752618f, 0.1651644f}, {0.04204986f, 0.005474207f, 0.1927065f}, {0.04871256f, 0.006285034f, 0.2236782f}, {0.05612868f, 0.007188068f, 0.2582109f},
    {0.06429866f, 0.008181786f, 0.2963632f}, {0.07319818f, 0.009260417f, 0.3381018f}, {0.08277331f, 0.01041303f, 0.3832822f}, {0.09295327f, 0.01162642f, 0.4316884f},
    {0.1037137f, 0.01289884f, 0.483244f}, {0.115052f, 0.01423442f, 0.5379345f}, {0.1269771f, 0.0156408f, 0.595774f}, {0.1395127f, 0.01712968f, 0.6568187f},
    {0.1526661f, 0.01871265f, 0.7210459f}, {0.1663054f, 0.02038394f, 0.7878635f}, {0.1802197f, 0.02212935f, 0.8563391f}, {0.1941448f, 0.02392985f, 0.9253017f},
    {0.2077647f, 0.02576133f, 0.9933444f}, {0.2207911f, 0.02760156f, 1.059178f}, {0.2332355f, 0.02945513f, 1.122832f}, {0.2452462f, 0.03133884f, 1.184947f},
    {0.2570397f, 0.03327575f, 1.246476f}, {0.2688989f, 0.03529554f, 1.308674f}, {0.2810677f, 0.03742705f, 1.372628f}, {0.2933967f, 0.03967137f, 1.437661f},
    {0.3055933f, 0.04201998f, 1.502449f}, {0.3173165f, 0.04446166f, 1.565456f}, {0.3281798f, 0.04698226f, 1.62494f}, {0.3378678f, 0.04956742f, 1.679488f},
    {0.3465097f, 0.05221219f, 1.729668f}, {0.3543953f, 0.05491387f, 1.776755f}, {0.3618655f, 0.05766919f, 1.822228f}, {0.3693084f, 0.06047429f, 1.867751f},
    {0.3770107f, 0.06332195f, 1.914504f}, {0.384685f, 0.06619271f, 1.961055f}, {0.3918591f, 0.06906185f, 2.005136f}, {0.3980192f, 0.0719019f, 2.044296f},
    {0.4026189f, 0.07468288f, 2.075946f}, {0.4052637f, 0.07738452f, 2.098231f}, {0.4062482f, 0.08003601f, 2.112591f}, {0.406066f, 0.08268524f, 2.121427f},
    {0.4052283f, 0.08538745f, 2.127239f}, {0.4042529f, 0.08820537f, 2.132574f}, {0.4034808f, 0.09118925f, 2.139093f}, {0.4025362f, 0.09431041f, 2.144815f},
    {0.4008675f, 0.09751346f, 2.146832f}, {0.3979327f, 0.1007349f, 2.14225f}, {0.3932139f, 0.103903f, 2.128264f}, {0.3864108f, 0.1069639f, 2.103205f},
    {0.3779513f, 0.1099676f, 2.069388f}, {0.3684176f, 0.1129992f, 2.03003f}, {0.3583473f, 0.1161541f, 1.988178f}, {0.3482214f, 0.1195389f, 1.946651f},
    {0.338383f, 0.1232503f, 1.907521f}, {0.3288309f, 0.1273047f, 1.870689f}, {0.3194977f, 0.1316964f, 1.835578f}, {0.3103345f, 0.1364178f, 1.801657f},
    {0.3013112f, 0.1414586f, 1.76844f}, {0.2923754f, 0.1468003f, 1.735338f}, {0.2833273f, 0.1524002f, 1.701254f}, {0.2739463f, 0.1582021f, 1.665053f},
    {0.2640352f, 0.16414f, 1.625712f}, {0.2534221f, 0.1701373f, 1.582342f}, {0.2420135f, 0.1761233f, 1.534439f}, {0.2299346f, 0.1820896f, 1.482544f},
    {0.2173617f, 0.1880463f, 1.427438f}, {0.2044672f, 0.1940065f, 1.369876f}, {0.1914176f, 0.1999859f, 1.310576f}, {0.1783672f, 0.2060054f, 1.250226f},
    {0.1654407f, 0.2120981f, 1.189511f}, {0.1527391f, 0.2183041f, 1.12905f}, {0.1403439f, 0.2246686f, 1.069379f}, {0.1283167f, 0.2312426f, 1.010952f},
    {0.1167124f, 0.2380741f, 0.9541809f}, {0.1056121f, 0.2451798f, 0.8995253f}, {0.09508569f, 0.2525682f, 0.847372f}, {0.08518206f, 0.2602479f, 0.7980093f},
    {0.0759312f, 0.2682271f, 0.7516389f}, {0.06733159f, 0.2765005f, 0.7082645f}, {0.05932018f, 0.2850035f, 0.6673867f}, {0.05184106f, 0.2936475f, 0.6284798f},
    {0.04486119f, 0.3023319f, 0.5911174f}, {0.0383677f, 0.3109438f, 0.5549619f}, {0.03237296f, 0.3194105f, 0.5198843f}, {0.02692095f, 0.3278683f, 0.4862772f},
    {0.0220407f, 0.3365263f, 0.4545497f}, {0.01773951f, 0.3456176f, 0.4249955f}, {0.01400745f, 0.3554018f, 0.3978114f}, {0.01082291f, 0.3660893f, 0.3730218f},
    {0.008168996f, 0.3775857f, 0.3502618f}, {0.006044623f, 0.389696f, 0.3291407f}, {0.004462638f, 0.4021947f, 0.3093356f}, {0.00344681f, 0.4148227f, 0.2905816f},
    {0.003009513f, 0.4273539f, 0.2726773f}, {0.003090744f, 0.4398206f, 0.2555143f}, {0.003611221f, 0.452336f, 0.2390188f}, {0.004491435f, 0.4650298f, 0.2231335f},
    {0.005652072f, 0.4780482f, 0.2078158f}, {0.007035322f, 0.4915173f, 0.1930407f}, {0.008669631f, 0.5054224f, 0.1788089f}, {0.01060755f, 0.5197057f, 0.1651287f},
    {0.01290468f, 0.5343012f, 0.1520103f}, {0.01561956f, 0.5491344f, 0.1394643f}, {0.0188164f, 0.5641302f, 0.1275353f}, {0.02256923f, 0.5792416f, 0.1163771f},
    {0.02694456f, 0.5944264f, 0.1061161f}, {0.0319991f, 0.6096388f, 0.09682266f}, {0.03778185f, 0.6248296f, 0.08852389f}, {0.04430635f, 0.6399656f, 0.08118263f},
    {0.05146516f, 0.6550943f, 0.07463132f}, {0.05912224f, 0.6702903f, 0.06870644f}, {0.0671422f, 0.6856375f, 0.06327834f}, {0.07538941f, 0.7012292f, 0.05824484f},
    {0.08376697f, 0.7171103f, 0.05353812f}, {0.09233581f, 0.7330917f, 0.04914863f}, {0.101194f, 0.7489041f, 0.04507511f}, {0.1104362f, 0.764253f, 0.04131175f},
    {0.1201511f, 0.7788199f, 0.03784916f}, {0.130396f, 0.792341f, 0.03467234f}, {0.141131f, 0.804851f, 0.03175471f}, {0.1522944f, 0.8164747f, 0.02907029f},
    {0.1638288f, 0.827352f, 0.02659651f}, {0.1756832f, 0.8376358f, 0.02431375f}, {0.1878114f, 0.8474653f, 0.02220677f}, {0.2001621f, 0.8568868f, 0.02026852f},
    {0.2126822f, 0.8659242f, 0.01849246f}, {0.2253199f, 0.8746041f, 0.01687084f}, {0.2380254f, 0.8829552f, 0.01539505f}, {0.2507787f, 0.8910274f, 0.0140545f},
    {0.2636778f, 0.8989495f, 0.01283354f}, {0.2768607f, 0.9068753f, 0.01171754f}, {0.2904792f, 0.9149652f, 0.01069415f}, {0.3046991f, 0.9233858f, 0.009753f},
    {0.3196485f, 0.9322325f, 0.008886096f}, {0.3352447f, 0.9412862f, 0.008089323f}, {0.351329f, 0.9502378f, 0.007359131f}, {0.3677148f, 0.9587647f, 0.006691736f},
    {0.3841856f, 0.9665325f, 0.006083223f}, {0.4005312f, 0.9732504f, 0.005529423f}, {0.4166669f, 0.9788415f, 0.005025504f}, {0.432542f, 0.9832867f, 0.004566879f},
    {0.4481063f, 0.986572f, 0.004149405f}, {0.4633109f, 0.9886887f, 0.003769336f}, {0.478144f, 0.9897056f, 0.003423302f}, {0.4927483f, 0.9899849f, 0.003108313f},
    {0.5073315f, 0.9899624f, 0.00282165f}, {0.5221315f, 0.9900731f, 0.00256083f}, {0.537417f, 0.99075f, 0.002323578f}, {0.5534217f, 0.9922826f, 0.002107847f},
    {0.5701242f, 0.9943837f, 0.001911867f}, {0.5874093f, 0.9966221f, 0.001734006f}, {0.6051269f, 0.9985649f, 0.001572736f}, {0.6230892f, 0.9997775f, 0.001426627f},
    {0.6410999f, 0.999944f, 0.001294325f}, {0.6590659f, 0.99922f, 0.001174475f}, {0.6769436f, 0.9978793f, 0.001065842f}, {0.6947143f, 0.9961934f, 0.0009673215f},
    {0.7123849f, 0.9944304f, 0.0008779264f}, {0.7299978f, 0.9927831f, 0.0007967847f}, {0.7476478f, 0.9911578f, 0.0007231502f}, {0.765425f, 0.9893925f, 0.0006563501f},
    {0.7834009f, 0.9873288f, 0.0005957678f}, {0.8016277f, 0.9848127f, 0.0005408385f}, {0.8201041f, 0.9817253f, 0.0004910441f}, {0.8386843f, 0.9780714f, 0.0004459046f},
    {0.8571936f, 0.973886f, 0.0004049826f}, {0.8754652f, 0.9692028f, 0.0003678818f}, {0.8933408f, 0.9640545f, 0.0003342429f}, {0.9106772f, 0.9584409f, 0.0003037407f},
    {0.9273554f, 0.9522379f, 0.0002760809f}, {0.9432502f, 0.9452968f, 0.000250997f}, {0.9582244f, 0.9374773f, 0.0002282474f}, {0.9721304f, 0.9286495f, 0.0002076129f},
    {0.9849237f, 0.9187953f, 0.0001888948f}, {0.9970067f, 0.9083014f, 0.0001719127f}, {1.008907f, 0.8976352f, 0.000156503f}, {1.021163f, 0.8872401f, 0.0001425177f},
    {1.034327f, 0.877536f, 0.000129823f}, {1.048753f, 0.868792f, 0.0001182974f}, {1.063937f, 0.8607474f, 0.000107831f}, {1.079166f, 0.8530233f, 9.832455E-05f},
    {1.093723f, 0.8452535f, 8.968787E-05f}, {1.106886f, 0.8370838f, 8.183954E-05f}, {1.118106f, 0.8282409f, 7.470582E-05f}, {1.127493f, 0.818732f, 6.821991E-05f},
    {1.135317f, 0.8086352f, 6.232132E-05f}, {1.141838f, 0.7980296f, 5.695534E-05f}, {1.147304f, 0.786995f, 5.207245E-05f}, {1.151897f, 0.775604f, 4.762781E-05f},
    {1.155582f, 0.7638996f, 4.358082E-05f}, {1.158284f, 0.7519157f, 3.989468E-05f}, {1.159934f, 0.7396832f, 3.653612E-05f}, {1.160477f, 0.7272309f, 3.347499E-05f},
    {1.15989f, 0.7145878f, 3.0684E-05f}, {1.158259f, 0.7017926f, 2.813839E-05f}, {1.155692f, 0.6888866f, 2.581574E-05f}, {1.152293f, 0.6759103f, 2.369574E-05f},
    {1.148163f, 0.6629035f, 2.175998E-05f}, {1.143345f, 0.6498911f, 1.999179E-05f}, {1.137685f, 0.636841f, 1.837603E-05f}, {1.130993f, 0.6237092f, 1.689896E-05f},
    {1.123097f, 0.6104541f, 1.554815E-05f}, {1.113846f, 0.5970375f, 1.431231E-05f}, {1.103152f, 0.5834395f, 1.318119E-05f}, {1.091121f, 0.5697044f, 1.214548E-05f},
    {1.077902f, 0.5558892f, 1.119673E-05f}, {1.063644f, 0.5420475f, 1.032727E-05f}, {1.048485f, 0.5282296f, 9.53013E-06f}, {1.032546f, 0.5144746f, 8.798979E-06f},
    {1.01587f, 0.5007881f, 8.128065E-06f}, {0.9984859f, 0.4871687f, 7.51216E-06f}, {0.9804227f, 0.473616f, 6.946506E-06f}, {0.9617111f, 0.4601308f, 6.426776E-06f},
    {0.9424119f, 0.446726f, 0.0f}, {0.9227049f, 0.4334589f, 0.0f}, {0.9027804f, 0.4203919f, 0.0f}, {0.8828123f, 0.407581f, 0.0f}, {0.8629581f, 0.3950755f, 0.0f},
    {0.8432731f, 0.3828894f, 0.0f}, {0.8234742f, 0.370919f, 0.0f}, {0.8032342f, 0.3590447f, 0.0f}, {0.7822715f, 0.3471615f, 0.0f}, {0.7603498f, 0.3351794f, 0.0f},
    {0.7373739f, 0.3230562f, 0.0f}, {0.713647f, 0.3108859f, 0.0f}, {0.6895336f, 0.298784f, 0.0f}, {0.6653567f, 0.2868527f, 0.0f}, {0.6413984f, 0.2751807f, 0.0f},
    {0.6178723f, 0.2638343f, 0.0f}, {0.5948484f, 0.252833f, 0.0f}, {0.57236f, 0.2421835f, 0.0f}, {0.5504353f, 0.2318904f, 0.0f}, {0.5290979f, 0.2219564f, 0.0f},
    {0.5083728f, 0.2123826f, 0.0f}, {0.4883006f, 0.2031698f, 0.0f}, {0.4689171f, 0.1943179f, 0.0f}, {0.4502486f, 0.185825f, 0.0f}, {0.4323126f, 0.1776882f, 0.0f},
    {0.415079f, 0.1698926f, 0.0f}, {0.3983657f, 0.1623822f, 0.0f}, {0.3819846f, 0.1550986f, 0.0f}, {0.3657821f, 0.1479918f, 0.0f}, {0.3496358f, 0.1410203f, 0.0f},
    {0.3334937f, 0.1341614f, 0.0f}, {0.3174776f, 0.1274401f, 0.0f}, {0.3017298f, 0.1208887f, 0.0f}, {0.2863684f, 0.1145345f, 0.0f}, {0.27149f, 0.1083996f, 0.0f},
    {0.2571632f, 0.1025007f, 0.0f}, {0.2434102f, 0.09684588f, 0.0f}, {0.2302389f, 0.09143944f, 0.0f}, {0.2176527f, 0.08628318f, 0.0f}, {0.2056507f, 0.08137687f, 0.0f},
    {0.1942251f, 0.07671708f, 0.0f}, {0.183353f, 0.07229404f, 0.0f}, {0.1730097f, 0.06809696f, 0.0f}, {0.1631716f, 0.06411549f, 0.0f}, {0.1538163f, 0.06033976f, 0.0f},
    {0.144923f, 0.05676054f, 0.0f}, {0.1364729f, 0.05336992f, 0.0f}, {0.1284483f, 0.05016027f, 0.0f}, {0.120832f, 0.04712405f, 0.0f}, {0.1136072f, 0.04425383f, 0.0f},
    {0.1067579f, 0.04154205f, 0.0f}, {0.1002685f, 0.03898042f, 0.0f}, {0.09412394f, 0.03656091f, 0.0f}, {0.08830929f, 0.03427597f, 0.0f}, {0.0828101f, 0.03211852f, 0.0f},
    {0.07761208f, 0.03008192f, 0.0f}, {0.07270064f, 0.02816001f, 0.0f}, {0.06806167f, 0.02634698f, 0.0f}, {0.06368176f, 0.02463731f, 0.0f}, {0.05954815f, 0.02302574f, 0.0f},
    {0.05564917f, 0.02150743f, 0.0f}, {0.05197543f, 0.02007838f, 0.0f}, {0.04851788f, 0.01873474f, 0.0f}, {0.04526737f, 0.01747269f, 0.0f}, {0.04221473f, 0.01628841f, 0.0f},
    {0.03934954f, 0.01517767f, 0.0f}, {0.0366573f, 0.01413473f, 0.0f}, {0.03412407f, 0.01315408f, 0.0f}, {0.03173768f, 0.01223092f, 0.0f}, {0.02948752f, 0.01136106f, 0.0f},
    {0.02736717f, 0.0105419f, 0.0f}, {0.02538113f, 0.00977505f, 0.0f}, {0.02353356f, 0.009061962f, 0.0f}, {0.02182558f, 0.008402962f, 0.0f}, {0.0202559f, 0.007797457f, 0.0f},
    {0.01881892f, 0.00724323f, 0.0f}, {0.0174993f, 0.006734381f, 0.0f}, {0.01628167f, 0.006265001f, 0.0f}, {0.01515301f, 0.005830085f, 0.0f}, {0.0141023f, 0.005425391f, 0.0f},
    {0.01312106f, 0.005047634f, 0.0f}, {0.01220509f, 0.00469514f, 0.0f}, {0.01135114f, 0.004366592f, 0.0f}, {0.01055593f, 0.004060685f, 0.0f}, {0.009816228f, 0.00377614f, 0.0f},
    {0.009128517f, 0.003511578f, 0.0f}, {0.008488116f, 0.003265211f, 0.0f}, {0.007890589f, 0.003035344f, 0.0f}, {0.007332061f, 0.002820496f, 0.0f},
    {0.006809147f, 0.002619372f, 0.0f}, {0.006319204f, 0.00243096f, 0.0f}, {0.005861036f, 0.002254796f, 0.0f}, {0.005433624f, 0.002090489f, 0.0f},
    {0.005035802f, 0.001937586f, 0.0f}, {0.004666298f, 0.001795595f, 0.0f}, {0.00432375f, 0.001663989f, 0.0f}, {0.004006709f, 0.001542195f, 0.0f},
    {0.003713708f, 0.001429639f, 0.0f}, {0.003443294f, 0.001325752f, 0.0f}, {0.003194041f, 0.00122998f, 0.0f}, {0.002964424f, 0.001141734f, 0.0f},
    {0.002752492f, 0.001060269f, 0.0f}, {0.002556406f, 0.0009848854f, 0.0f}, {0.002374564f, 0.0009149703f, 0.0f}, {0.002205568f, 0.0008499903f, 0.0f},
    {0.002048294f, 0.0007895158f, 0.0f}, {0.001902113f, 0.0007333038f, 0.0f}, {0.001766485f, 0.0006811458f, 0.0f}, {0.001640857f, 0.0006328287f, 0.0f},
    {0.001524672f, 0.0005881375f, 0.0f}, {0.001417322f, 0.0005468389f, 0.0f}, {0.001318031f, 0.0005086349f, 0.0f}, {0.001226059f, 0.0004732403f, 0.0f},
    {0.001140743f, 0.0004404016f, 0.0f}, {0.001061495f, 0.0004098928f, 0.0f}, {0.0009877949f, 0.0003815137f, 0.0f}, {0.0009191847f, 0.0003550902f, 0.0f},
    {0.0008552568f, 0.0003304668f, 0.0f}, {0.0007956433f, 0.000307503f, 0.0f}, {0.000740012f, 0.0002860718f, 0.0f}, {0.000688098f, 0.0002660718f, 0.0f},
    {0.0006397864f, 0.0002474586f, 0.0f}, {0.0005949726f, 0.0002301919f, 0.0f}, {0.0005535291f, 0.0002142225f, 0.0f}, {0.0005153113f, 0.0001994949f, 0.0f},
    {0.0004801234f, 0.0001859336f, 0.0f}, {0.0004476245f, 0.0001734067f, 0.0f}, {0.0004174846f, 0.0001617865f, 0.0f}, {0.0003894221f, 0.0001509641f, 0.0f},
    {0.0003631969f, 0.0001408466f, 0.0f}, {0.0003386279f, 0.0001313642f, 0.0f}, {0.0003156452f, 0.0001224905f, 0.0f}, {0.0002941966f, 0.000114206f, 0.0f},
    {0.0002742235f, 0.0001064886f, 0.0f}, {0.0002556624f, 9.931439E-05f, 0.0f}, {0.000238439f, 9.265512E-05f, 0.0f}, {0.0002224525f, 8.647225E-05f, 0.0f},
    {0.0002076036f, 8.07278E-05f, 0.0f}, {0.0001938018f, 7.538716E-05f, 0.0f}, {0.0001809649f, 7.041878E-05f, 0.0f}, {0.0001690167f, 6.579338E-05f, 0.0f},
    {0.0001578839f, 6.14825E-05f, 0.0f}, {0.0001474993f, 5.746008E-05f, 0.0f}, {0.0001378026f, 5.370272E-05f, 0.0f}, {0.0001287394f, 5.018934E-05f, 0.0f},
    {0.0001202644f, 4.690245E-05f, 0.0f}, {0.0001123502f, 4.383167E-05f, 0.0f}, {0.0001049725f, 4.09678E-05f, 0.0f}, {9.810596E-05f, 3.830123E-05f, 0.0f},
    {9.172477E-05f, 3.582218E-05f, 0.0f}, {8.579861E-05f, 3.351903E-05f, 0.0f}, {8.028174E-05f, 3.137419E-05f, 0.0f}, {7.513013E-05f, 2.937068E-05f, 0.0f},
    {7.030565E-05f, 2.74938E-05f, 0.0f}, {6.577532E-05f, 2.573083E-05f, 0.0f}, {6.151508E-05f, 2.407249E-05f, 0.0f}, {5.752025E-05f, 2.251704E-05f, 0.0f},
    {5.378813E-05f, 2.10635E-05f, 0.0f}, {5.03135E-05f, 1.970991E-05f, 0.0f}, {4.708916E-05f, 1.845353E-05f, 0.0f}, {4.410322E-05f, 1.728979E-05f, 0.0f},
    {4.13315E-05f, 1.620928E-05f, 0.0f}, {3.874992E-05f, 1.520262E-05f, 0.0f}, {3.633762E-05f, 1.426169E-05f, 0.0f}, {3.407653E-05f, 1.337946E-05f, 0.0f},
    {3.195242E-05f, 1.255038E-05f, 0.0f}, {2.995808E-05f, 1.177169E-05f, 0.0f}, {2.808781E-05f, 1.104118E-05f, 0.0f}, {2.633581E-05f, 1.035662E-05f, 0.0f},
    {2.46963E-05f, 9.715798E-06f, 0.0f}, {2.316311E-05f, 9.116316E-06f, 0.0f}, {2.172855E-05f, 8.555201E-06f, 0.0f}, {2.038519E-05f, 8.029561E-06f, 0.0f},
    {1.912625E-05f, 7.536768E-06f, 0.0f}, {1.794555E-05f, 7.074424E-06f, 0.0f}, {1.683776E-05f, 6.640464E-06f, 0.0f}, {1.579907E-05f, 6.233437E-06f, 0.0f},
    {1.482604E-05f, 5.852035E-06f, 0.0f}, {1.391527E-05f, 5.494963E-06f, 0.0f}, {1.306345E-05f, 5.160948E-06f, 0.0f}, {1.22672E-05f, 4.848687E-06f, 0.0f},
    {1.152279E-05f, 4.556705E-06f, 0.0f}, {1.082663E-05f, 4.28358E-06f, 0.0f}, {1.01754E-05f, 4.027993E-06f, 0.0f}, {9.565993E-06f, 3.788729E-06f, 0.0f},
    {8.995405E-06f, 3.564599E-06f, 0.0f}, {8.460253E-06f, 3.354285E-06f, 0.0f}, {7.957382E-06f, 3.156557E-06f, 0.0f}, {7.483997E-06f, 2.970326E-06f, 0.0f},
    {7.037621E-06f, 2.794625E-06f, 0.0f}, {6.616311E-06f, 2.628701E-06f, 0.0f}, {6.219265E-06f, 2.472248E-06f, 0.0f}, {5.845844E-06f, 2.32503E-06f, 0.0f},
    {5.495311E-06f, 2.186768E-06f, 0.0f}, {5.166853E-06f, 2.057152E-06f, 0.0f}, {4.859511E-06f, 1.935813E-06f, 0.0f}, {4.571973E-06f, 1.822239E-06f, 0.0f},
    {4.30292E-06f, 1.715914E-06f, 0.0f}, {4.051121E-06f, 1.616355E-06f, 0.0f}, {3.815429E-06f, 1.523114E-06f, 0.0f}, {3.594719E-06f, 1.43575E-06f, 0.0f},
    {3.387736E-06f, 1.353771E-06f, 0.0f}, {3.193301E-06f, 1.276714E-06f, 0.0f}, {3.010363E-06f, 1.204166E-06f, 0.0f}, {2.83798E-06f, 1.135758E-06f, 0.0f},
    {2.675365E-06f, 1.071181E-06f, 0.0f}, {2.52202E-06f, 1.010243E-06f, 0.0f}, {2.377511E-06f, 9.527779E-07f, 0.0f}, {2.241417E-06f, 8.986224E-07f, 0.0f},
    {2.113325E-06f, 8.476168E-07f, 0.0f}, {1.99283E-06f, 7.996052E-07f, 0.0f}, {1.879542E-06f, 7.544361E-07f, 0.0f}, {1.773083E-06f, 7.119624E-07f, 0.0f},
    {1.673086E-06f, 6.720421E-07f, 0.0f}, {1.579199E-06f, 6.34538E-07f, 0.0f}};

  return kCIE2006[i];
}

ETX_GPU_CODE float3 xyz_to_rgb(const float3& xyz) {
  return {
    3.24045420f * xyz.x - 1.5371385f * xyz.y - 0.4985314f * xyz.z,
    -0.9692660f * xyz.x + 1.8760108f * xyz.y + 0.0415560f * xyz.z,
    0.05564340f * xyz.x - 0.2040259f * xyz.y + 1.0572252f * xyz.z,
  };
}

ETX_GPU_CODE float3 rgb_to_xyz(const float3& rgb) {
  return {
    0.4124564f * rgb.x + 0.3575760f * rgb.y + 0.1804375f * rgb.z,
    0.2126729f * rgb.x + 0.7151521f * rgb.y + 0.0721750f * rgb.z,
    0.0193339f * rgb.x + 0.1191920f * rgb.y + 0.9503041f * rgb.z,
  };
}

ETX_GPU_CODE float4 rgb_to_xyz4(const float3& rgb) {
  return {
    0.412453f * rgb.x + 0.357580f * rgb.y + 0.180423f * rgb.z,
    0.212671f * rgb.x + 0.715160f * rgb.y + 0.072169f * rgb.z,
    0.019334f * rgb.x + 0.119193f * rgb.y + 0.950227f * rgb.z,
    1.0f,
  };
}

ETX_GPU_CODE float black_body_radiation_maximum_wavelength(float t_kelvins) {
  return 2.8977729e+6f / t_kelvins;
}

ETX_GPU_CODE float black_body_radiation(float wavelength_nm, float t_kelvins) {
  ETX_ASSERT(t_kelvins > 0);

  // wavelength (in nm) is scaled to reduce floating point errors, constants are scaled correspondingly
  constexpr float wavelengt_scale = 1.0f / 1000.0f;
  constexpr float Lc1 = 3.7417712e+5f;  // 2 * pi * h * c * c * (10^21 - from wavelength scale)
  constexpr float Lc2 = 1.4387752e+4f;  // h * c / k * (10^-6 - from wavelength scale)

  wavelength_nm *= wavelengt_scale;
  float wl5 = wavelength_nm * (wavelength_nm * wavelength_nm) * (wavelength_nm * wavelength_nm);

  float e0 = expf(Lc2 / (wavelength_nm * t_kelvins));
  ETX_ASSERT(isnan(e0) == false);

  float d = wl5 * (e0 - 1.0f);
  ETX_ASSERT(isnan(d) == false);

  return isinf(d) ? 0.0f : (Lc1 / d);
}

constexpr const float kYIntegral() {
  float result = 0.0f;
  for (uint32_t i = 0; i < WavelengthCount; ++i) {
    result += spectral_xyz(i).y;
  }
  return result;
}

}  // namespace spectrum

struct SpectralQuery {
  enum : uint32_t {
    Spectral = 1u << 0u,
  };
  float wavelength = spectrum::kUndefinedWavelength;
  uint32_t flags = 0u;

  SpectralQuery() = default;

  SpectralQuery(float w, const uint32_t& f)
    : wavelength(w)
    , flags(f) {
  }

  bool spectral() const {
    return (flags & Spectral) != 0;
  };

  float sampling_pdf() const {
    return spectral() ? (0.0039398042f / sqr(std::cosh(0.0072f * (wavelength - 538.0f)))) : 1.0f;
  }

  bool valid() const {
    return (wavelength >= spectrum::kShortestWavelength) && (wavelength <= spectrum::kLongestWavelength);
  }

  static SpectralQuery sample() {
    return SpectralQuery{
      spectrum::kUndefinedWavelength,
      0u,
    };
  }

  static SpectralQuery spectral_sample(float rnd) {
    constexpr auto offset = 0x1.35ce7a0000000p-5f;
    constexpr auto scale = 1.0f - offset;
    float w = 538.0f - 138.888889f * std::atanh(0.85691062f - 1.82750197f * (rnd * scale + offset));
    return SpectralQuery{w, Spectral};
  }
};

struct SpectralResponse : public SpectralQuery {
  float value = 0.0f;
  float3 integrated = {};

  SpectralResponse() = default;

  SpectralResponse(const SpectralQuery q)
    : SpectralQuery(q) {
  }

  SpectralResponse(const SpectralQuery q, float a)
    : SpectralQuery(q)
    , integrated{a, a, a}
    , value(a) {
  }

  SpectralResponse(const SpectralQuery q, const float3& c)
    : SpectralQuery(q)
    , integrated{c} {
  }

  float component_count() const {
    return spectral() ? 1.0f : 3.0f;
  }

  const SpectralQuery& query() const {
    return *this;
  }

  ETX_GPU_CODE float3 to_xyz() const {
    if (spectral() == false) {
      return spectrum::rgb_to_xyz(integrated);
    }

    if ((value == 0.0f) || (wavelength < spectrum::kShortestWavelength) || (wavelength > spectrum::kLongestWavelength))
      return {};

    constexpr float kYScale = 1.0f / spectrum::kYIntegral();

    ETX_ASSERT(valid());
    float w = floorf(wavelength);
    float dw = wavelength - w;
    uint32_t i = static_cast<uint32_t>(w - spectrum::kShortestWavelength);
    uint32_t j = min(i + 1u, spectrum::WavelengthCount - 1u);
    float3 xyz0 = spectrum::spectral_xyz(i);
    float3 xyz1 = spectrum::spectral_xyz(j);
    return lerp<float3>(xyz0, xyz1, dw) * (value * kYScale);
  }

  ETX_GPU_CODE float3 to_rgb() const {
    return spectral() ? spectrum::xyz_to_rgb(to_xyz()) : integrated;
  }

  ETX_GPU_CODE float minimum() const {
    return spectral() ? value : min(integrated.x, min(integrated.y, integrated.z));
  }

  ETX_GPU_CODE float maximum() const {
    return spectral() ? value : max(integrated.x, max(integrated.y, integrated.z));
  }

  ETX_GPU_CODE float monochromatic() const {
    return spectral() ? value : luminance(integrated);
  }

  ETX_GPU_CODE float sum() const {
    return spectral() ? value : integrated.x + integrated.y + integrated.z;
  }

  ETX_GPU_CODE float average() const {
    return spectral() ? value : (integrated.x + integrated.y + integrated.z) / 3.0f;
  }

  ETX_GPU_CODE float component(uint32_t i) const {
    ETX_ASSERT(i < 3);
    return spectral() ? value : *(&integrated.x + i);
  }

  ETX_GPU_CODE bool valid() const {
    return spectral() ? valid_value(value) : valid_value(integrated);
  }

  ETX_GPU_CODE bool is_zero() const {
    return spectral() ? (value <= kEpsilon) : (integrated.x <= kEpsilon) && (integrated.y <= kEpsilon) && (integrated.z <= kEpsilon);
  }

#define SPECTRAL_OP(OP)                                                        \
  ETX_GPU_CODE SpectralResponse& operator OP(const SpectralResponse & other) { \
    ETX_ASSERT_EQUAL(wavelength, other.wavelength);                            \
    integrated OP other.integrated;                                            \
    value OP other.value;                                                      \
    return *this;                                                              \
  }
  SPECTRAL_OP(+=)
  SPECTRAL_OP(-=)
  SPECTRAL_OP(*=)
  SPECTRAL_OP(/=)
#undef SPECTRAL_OP

#define SPECTRAL_OP(OP)                                                                                                              \
  ETX_GPU_CODE SpectralResponse operator OP(const SpectralResponse& other) const {                                                   \
    ETX_ASSERT_EQUAL(wavelength, other.wavelength);                                                                                  \
    ETX_ASSERT((spectral() && other.spectral()) || ((spectral() == false) && (other.spectral() == false)));                          \
    return spectral() ? SpectralResponse{query(), value OP other.value} : SpectralResponse{query(), integrated OP other.integrated}; \
  }
  SPECTRAL_OP(+)
  SPECTRAL_OP(-)
  SPECTRAL_OP(*)
  SPECTRAL_OP(/)
#undef SPECTRAL_OP

#define SPECTRAL_OP(OP)                                     \
  ETX_GPU_CODE SpectralResponse& operator OP(float other) { \
    integrated OP other;                                    \
    value OP other;                                         \
    return *this;                                           \
  }
  SPECTRAL_OP(+=)
  SPECTRAL_OP(-=)
  SPECTRAL_OP(*=)
  SPECTRAL_OP(/=)
#undef SPECTRAL_OP

#define SPECTRAL_OP(OP)                                                                                             \
  ETX_GPU_CODE SpectralResponse operator OP(float other) const {                                                    \
    return spectral() ? SpectralResponse{query(), value OP other} : SpectralResponse{query(), integrated OP other}; \
  }
  SPECTRAL_OP(+)
  SPECTRAL_OP(-)
  SPECTRAL_OP(*)
  SPECTRAL_OP(/)
#undef SPECTRAL_OP
};

ETX_GPU_CODE SpectralResponse operator*(float other, const SpectralResponse& s) {
  return s * other;
}
ETX_GPU_CODE SpectralResponse operator/(float other, const SpectralResponse& s) {
  return s.spectral() ? SpectralResponse{s.query(), other / s.value} : SpectralResponse{s.query(), other / s.integrated};
}
ETX_GPU_CODE SpectralResponse operator+(float other, const SpectralResponse& s) {
  return s + other;
}
ETX_GPU_CODE SpectralResponse operator-(const SpectralResponse& s) {
  return s.spectral() ? SpectralResponse{s.query(), -s.value} : SpectralResponse{s.query(), -s.integrated};
}
ETX_GPU_CODE SpectralResponse operator-(float other, const SpectralResponse& s) {
  return s.spectral() ? SpectralResponse{s.query(), other - s.value} : SpectralResponse{s.query(), other - s.integrated};
}
ETX_GPU_CODE SpectralResponse exp(const SpectralResponse& s) {
  return s.spectral() ? SpectralResponse{s.query(), expf(s.value)} : SpectralResponse{s.query(), exp(s.integrated)};
}
ETX_GPU_CODE SpectralResponse sqrt(const SpectralResponse& s) {
  return s.spectral() ? SpectralResponse{s.query(), sqrtf(s.value)} : SpectralResponse{s.query(), sqrt(s.integrated)};
}
ETX_GPU_CODE SpectralResponse cos(const SpectralResponse& s) {
  return s.spectral() ? SpectralResponse{s.query(), cosf(s.value)} : SpectralResponse{s.query(), cos(s.integrated)};
}
ETX_GPU_CODE SpectralResponse abs(const SpectralResponse& s) {
  return s.spectral() ? SpectralResponse{s.query(), fabsf(s.value)} : SpectralResponse{s.query(), abs(s.integrated)};
}
ETX_GPU_CODE SpectralResponse saturate(const SpectralResponse& s) {
  return s.spectral() ? SpectralResponse{s.query(), saturate(s.value)} : SpectralResponse{s.query(), saturate(s.integrated)};
}
ETX_GPU_CODE SpectralResponse sign(const SpectralResponse& b) {
  return b.spectral() ? SpectralResponse{b.query(), sign(b.value)} : SpectralResponse(b.query(), sign(b.integrated));
}
ETX_GPU_CODE SpectralResponse atan(const SpectralResponse& b) {
  return b.spectral() ? SpectralResponse{b.query(), atanf(b.value)} : SpectralResponse(b.query(), atan(b.integrated));
}
ETX_GPU_CODE SpectralResponse pow(const SpectralResponse& a, float b) {
  return a.spectral() ? SpectralResponse{a.query(), powf(a.value, b)} : SpectralResponse(a.query(), pow(a.integrated, b));
}
ETX_GPU_CODE SpectralResponse pow(const SpectralResponse& a, const SpectralResponse& b) {
  return a.spectral() ? SpectralResponse{b.query(), powf(a.value, b.value)} : SpectralResponse(b.query(), pow(a.integrated, b.integrated));
}
ETX_GPU_CODE SpectralResponse max(const SpectralResponse& a, float b) {
  return a.spectral() ? SpectralResponse{a.query(), fmaxf(a.value, b)} : SpectralResponse(a.query(), max(a.integrated, b));
}
ETX_GPU_CODE SpectralResponse max(float a, const SpectralResponse& b) {
  return b.spectral() ? SpectralResponse{b.query(), fmaxf(b.value, a)} : SpectralResponse(b.query(), max(b.integrated, a));
}
ETX_GPU_CODE SpectralResponse min(const SpectralResponse& a, float b) {
  return a.spectral() ? SpectralResponse{a.query(), fminf(a.value, b)} : SpectralResponse(a.query(), min(a.integrated, b));
}
ETX_GPU_CODE SpectralResponse min(float a, const SpectralResponse& b) {
  return b.spectral() ? SpectralResponse{b.query(), fminf(b.value, a)} : SpectralResponse(b.query(), min(b.integrated, a));
}

ETX_GPU_CODE bool valid_value(const SpectralResponse& v) {
  return v.valid();
}

#if (ETX_DEBUG || ETX_FORCE_VALIDATION)
template <>
ETX_GPU_CODE void print_invalid_value<complex>(const char* name, const complex& z, const char* filename, uint32_t line) {
  printf("Validation failed: %s (%f + i * %f) at %s [%u]\n", name, z.real(), z.imag(), filename, line);
}

template <>
ETX_GPU_CODE void print_invalid_value<SpectralResponse>(const char* name, const SpectralResponse& v, const char* filename, uint32_t line) {
  printf("Validation failed: %s (%f : %f %f %f / %f) at %s [%u]\n", name, v.wavelength, v.integrated.x, v.integrated.y, v.integrated.z, v.value, filename, line);
}
#endif

struct Spectrums;

struct ETX_ALIGNED SpectralDistribution {
  constexpr static const float3 kRGBLuminanceScale = {0.817660332f, 1.05418909f, 1.09945524f};

  enum Class : uint32_t {
    Invalid,
    Reflectance,
    Conductor,
    Dielectric,
    Illuminant,
  };

  struct {
    float wavelength = 0.0f;
    float power = 0.0f;
  } spectral_entries[spectrum::WavelengthCount] = {};

  uint32_t spectral_entry_count = 0u;

 public:  // device
  ETX_GPU_CODE SpectralResponse query(const SpectralQuery q) const {
    if (q.spectral() == false) {
      return SpectralResponse{q, integrated_value};
    }

    ETX_ASSERT(q.valid());

    auto lower_bound = [this](float wavelength) {
      uint32_t b = 0;
      uint32_t e = spectral_entry_count;
      do {
        uint32_t m = b + (e - b) / 2;
        if (spectral_entries[m].wavelength > wavelength) {
          e = m;
        } else {
          b = m;
        }
      } while ((e - b) > 1);
      return b;
    };

    uint32_t i = lower_bound(q.wavelength);
    if (i >= spectral_entry_count) {
      return {q, 0.0f};
    }

    if ((i == 0) && (q.wavelength < spectral_entries[i].wavelength)) {
      return {q, 0.0f};
    }

    if ((i + 1 == spectral_entry_count) && (q.wavelength > spectral_entries[i].wavelength)) {
      return {q, 0.0f};
    }

    uint32_t j = min(i + 1u, spectral_entry_count - 1);
    float t = (i == j) ? 0.0f : (q.wavelength - spectral_entries[i].wavelength) / (spectral_entries[j].wavelength - spectral_entries[i].wavelength);
    float p = lerp(spectral_entries[i].power, spectral_entries[j].power, t);
    ETX_VALIDATE(p);
    return SpectralResponse{q, p};
  }

  ETX_GPU_CODE SpectralResponse operator()(const SpectralQuery q) const {
    return query(q);
  }

  ETX_GPU_CODE bool empty() const {
    return spectral_entry_count == 0;
  }

  ETX_GPU_CODE bool is_zero() const {
    for (uint32_t i = 0; i < spectral_entry_count; ++i) {
      if (spectral_entries[i].power != 0.0f) {
        return false;
      }
    }
    return true;
  }

 public:
  SpectralDistribution() = default;

  void scale(float factor);

  float3 integrate_to_xyz() const;

  const float3& integrated() const;

  float luminance() const;
  float maximum_spectral_power() const;

  bool valid() const;

  static SpectralDistribution from_samples(const float2 wavelengths_power[], uint64_t count);

  static SpectralDistribution null();
  static SpectralDistribution constant(float value);
  static SpectralDistribution from_black_body(float temperature, float scale);
  static SpectralDistribution from_normalized_black_body(float temperature, float scale);
  static SpectralDistribution rgb_reflectance(const float3& rgb);
  static SpectralDistribution rgb_luminance(const float3& rgb);

  static Class load_from_file(const char* file_name, SpectralDistribution& values0, SpectralDistribution* values1, bool extend_range);

 private:
  friend struct RefractiveIndex;
  float3 integrated_value = {};
};

struct RefractiveIndex {
  SpectralDistribution::Class cls = SpectralDistribution::Class::Invalid;
  SpectralDistribution eta;
  SpectralDistribution k;

  static RefractiveIndex load_from_file(const char* file_name);

  struct Sample : public SpectralQuery {
    SpectralDistribution::Class cls = SpectralDistribution::Class::Invalid;
    SpectralResponse eta;
    SpectralResponse k;

    ETX_GPU_CODE complex as_complex_x() const {
      ETX_ASSERT(spectral() == false);
      return complex{eta.integrated.x, k.integrated.x};
    }

    ETX_GPU_CODE complex as_complex_y() const {
      ETX_ASSERT(spectral() == false);
      return {eta.integrated.y, k.integrated.y};
    }

    ETX_GPU_CODE complex as_complex_z() const {
      ETX_ASSERT(spectral() == false);
      return {eta.integrated.z, k.integrated.z};
    }

    ETX_GPU_CODE complex as_complex() const {
      ETX_ASSERT(spectral());
      return {eta.value, k.value};
    }

    ETX_GPU_CODE complex as_monochromatic_complex() const {
      return {eta.monochromatic(), k.monochromatic()};
    }
  };

  ETX_GPU_CODE Sample at(SpectralQuery q) const {
    Sample result = {q};
    result.cls = cls;
    result.eta = eta.empty() ? SpectralResponse(q, 1.0f) : eta(q);
    result.k = k.empty() ? SpectralResponse(q, 0.0f) : k(q);
    return result;
  }

  ETX_GPU_CODE Sample operator()(const SpectralQuery q) const {
    return at(q);
  }
};

struct ETX_ALIGNED Spectrums {
  RefractiveIndex thinfilm = {};
  RefractiveIndex conductor = {};
  RefractiveIndex dielectric = {};
  SpectralDistribution rayleigh = {};
  SpectralDistribution mie = {};
  SpectralDistribution ozone = {};
  SpectralDistribution black = {};
};

SpectralResponse rgb_response(const SpectralQuery spect, const float3& rgb);

namespace spectrum {

Pointer<Spectrums> shared();

void init(Spectrums&);

}  // namespace spectrum

}  // namespace etx
